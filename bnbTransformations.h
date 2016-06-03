#ifndef BNB_TRANSFORMATIONS
#define BNB_TRANSFORMATIONS

#include <map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "accelerators/kdtree.h"
#include "4pcs.h"
#include <vector>

/** \brief This class efficiently finds the best candidate transformation that best aligns two point clouds.
  *  
  * Given two point clouds and a list of transformation, this class uses a branch and bound algorithm to 
  * efficiently keep track of the transformation that gives the best registration.  The quality of a 
  * transformation is measured using the Largest Common Overlap (LCP) measure which counts the number 
  * of source points that align with the target.  The higher the LCP the better the registration.
  * This class borrows its main idea from the paper titled GO-ICP: Globally Optimal Solution to 3D ICP Point-Set Registration
  * published in ICCV13 by Yang et al.  The following paragraph explains the algorithm for rotations only.
  *
  * A brute force approach would be to calculate the LCP measure for all candidate rotations.  However, this is not necessary.
  * The 3D rotation space can be represented using the SE(3) (Special Euclidean Group) using the angle-axis encoding
  * of a rotation matrix.  The special euclidean group forms a 3D bounded space which can be divided heirarchicaly using 
  * an octree.  Each node in this octree represents a subspace of rotations.  For each subspace, the best possible achievable
  * LCP measure using a transformation within that subspace can be efficiently calculated.  Given this, one can maintain
  * a global best upper bound achieved so far as well as local upper bounds for each subspace.  Once the global upper bound 
  * becomes better than a local subspace upper bound then all rotations that are part
  * of this subspace can be ignored.  Therefore their LCP measure doesn't have to be calculated.  This is where the efficiency is introduced.
  *
  * This class is designed to be used with registration algorithms that require verificatin of a 
  * large number of transformations (e.g RANSAC algorithms)
  * Example Usage:
  * \code
  *   bnbTransformations<3> *bnbRotations;
  *   bnbRotations = new bnbTransformations<3>(kd_tree_of_target_cloud, src_cloud_, epsilon);
  *   int bnb_depth = 7;
  *   bnbRotations->initForRotations(bnb_depth);
  *   bnbRotations->add(transform); // this operations is repated for all canidate transformations
  *   double best_lcp = bnbRotations->getBestLCP()
  * \endcode
  *
  * \author Mustafa Mohamad
*/
template <unsigned char DIM>
class bnbTransformations
{
public:
  /** \brief Constructor 
   *  \param[in] tgt_search the KdTree used to perform nn queries (used for LCP calculation)
   *  \param[in] src_cloud the source cloud to be registered to the target cloud
   *  \param[in] epsilon error tolerance used to determine whether a nearest neighbour is considered coincident
   */
  bnbTransformations(Super4PCS::KdTree<double> tgt_search, std::vector<match_4pcs::Point3D> src_cloud, double epsilon);

  /** \brief Destructor */
  virtual ~bnbTransformations();

  /** \breif Initialize the octree rigid transformation space representation 
   *  \param[in] tree_depth the maximum depth of the octree
   *  \param[in] max_translation the maximum translation in any of x,y,z directions
   */
  void init (int tree_depth, double max_translation);
  
  /* \brief Initialize the octree rigid rotation space representation
   * \parampin] tree_depth the maximum depth of the octree
   * default depth level is int(log_2(360)) = 8 
   */
  void initForRotations(int tree_depth=8);

  /** \brief Add a transformation to the rigid transformation space
   *  \param[in] T the transformation in homogeneous form
   */
  void add(Eigen::Matrix4d &T);

  /** \brief Return the best transformation of all added transformations */
  Eigen::Matrix4d getBestT();

  /** \brief Return the bestLCP of all added transformations */
  double getBestLCP();

  /** \brief Return number of nodes in the tree */
  void printTreeSize();

  /** \brief Print a textual representation of the transformation octree */
  void printTree();

  /** Return the number of LCP calculations  performed */
  void printNumLCPS();

protected:
  class MapNode; // forward declaration
  friend class MapNode;
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  /** \brief This class represents the node of the octree used to subdivide the rigid transformation space */
  class MapNode 
  {
    friend class bnbTransformations<DIM>;
  public:  
    MapNode() {}
    
    /** \brief Constructor
      * \param[in] lvl At which level in the tree is this node
      * \param[in] cell_bounds the bounds that limit the subspace that the node occupies
      */
    MapNode(int lvl, std::vector<double> &cell_bounds)
    : upper_bound_(std::numeric_limits<int>::max()),
      lvl_(lvl),
      cell_bounds_(cell_bounds)
    {
      for (int i = 0; i < DIM; ++i)
        centre_(i) = (cell_bounds[2*i] + cell_bounds[2*i+1]) / 2.0;
      angle_ = centre_.norm();
      axis_ = centre_.normalized();
    }

  protected:
    /** \brief Set centre of this node */
    void setCentre(double a[DIM])
    {
      for (int i = 0; i < DIM; ++i)
        centre_(i) = a[i];
    }
    
    /** \brief Compute the child cell that the Axis Angle transform would occupy relative
      * to the centre of the node
      * \param[in] aa the angle angle axis transform
      * \return the id of the child node (e.g for a 3D octree there are 8 possible children with ids 0 - 7)
      */
    char computeChildId(Eigen::AngleAxis<double> &aa)
    {
      char id = 0;
      Eigen::Vector3d aa_compact = aa.angle() * aa.axis();
      for (int i = 0; i < DIM; ++i)
      {
        // set dimentional bit to 1 if axis_angle[dim] >= centre[dim]
        if (aa_compact(i) >= centre_(i))
          id |= static_cast<unsigned char>(pow(2,i));
      }
      return id;
    }

    /** \brief Return a pointer to the child given its id
     *  \param[in] bnb the bnbTransformation class instance
     *  \param[in] id the id of the child
     *  \return a pointer to the child node
     */
    MapNode* getChild(bnbTransformations &bnb, char id)
    {
      if (children_.find(id) != children_.end())
        return &children_.at(id);
      else
      {
        bnb.node_count_++;
        std::vector<double> child_cell_bounds;
        computeChildBounds(id, child_cell_bounds);
        MapNode child(this->lvl_ + 1, child_cell_bounds);
        child.computeLCPUpperBound(bnb);
        children_[id] = child;
        return &children_.at(id);
      }
    }

    /** \brief Compute the dimensional bounds of a child node
      * \param[in] child_id used to specify which child
      * \param[out] child_cell_bounds stores the computed child dimensional bounds
      */
    void computeChildBounds(char child_id, std::vector<double> &child_cell_bounds)
    {
      child_cell_bounds.resize(DIM*2);
      for (char i = 0; i < DIM; ++i)
        // update the lower or upper bound of the dimension depending on whether the bit is set or not
        if (child_id & 1 << i)
        {
          child_cell_bounds[2*i] = centre_(i);
          child_cell_bounds[2*i+1] = cell_bounds_[2*i+1];
        }
        else
        {
          child_cell_bounds[2*i] = cell_bounds_[2*i];
          child_cell_bounds[2*i+1] = centre_(i);
        }
    }

    /** \brief Compute the LCP measure upper bound
      * \param[in] bnb the bnbTransformations instance this node is part of
      */
    void computeLCPUpperBound(bnbTransformations &bnb)
    {
      // TODO: Need to make this funtion work for 6D tranformations

      // calculate the error, delta, associated with this box
      // 1. distance from upper corner to centre
      double sqr_alpha, alpha, delta, sqr_error;
      for (int i = 0; i < DIM; ++i)
      {
        sqr_alpha = pow(cell_bounds_[2*i+1] - centre_(i),2);
      }
      alpha = sqrt(sqr_alpha);
      //cerr << "Node Level: "<< this->lvl_ << endl;
      //cerr << "cos alpha: " << cos(alpha) << endl;
      //cerr << "epsilon: " << bnb.epsilon_ << endl;
 
      Eigen::Matrix3d R;
      R = Eigen::AngleAxis<double>(angle_,axis_).toRotationMatrix();
      upper_bound_ = 0;
      Eigen::Vector3d query_point;
      for (int i = 0; i < bnb.src_cloud_.size(); ++i)
      {
        match_4pcs::Point3D q = bnb.src_cloud_[i];
        
        query_point << q.x,
                       q.y,
                       q.z;
        query_point = R * query_point;
        
        // calculate squared delta according to eqn 8 of paper
        delta = sqrt(2*query_point.squaredNorm()*(1-cos(alpha)));
        sqr_error = pow(bnb.epsilon_ + delta,2);
        
        // perform the query
        typename Super4PCS::KdTree<double>::Index resId =
        bnb.tgt_search_.doQueryRestrictedClosest(query_point, sqr_error);
        if ( resId != Super4PCS::KdTree<double>::invalidIndex() ) {
           upper_bound_++;
        }
      }
    }
  
    /* \brief centre of high dim voxel */
    Eigen::Matrix<double, DIM, 1> centre_;

    /* \brief Angle and Axis representation */
    double angle_;
    Eigen::Vector3d axis_;

    /* \brief Bounds of the voxel */
    std::vector<double> cell_bounds_;

    /* \brief LCP upper bound for the transformation represented by centre of this node */
    int upper_bound_;
    
    /* The children of this MapNode */
    std::map<unsigned char, MapNode> children_;

    /* depth of this node in the tree */
    int lvl_;
   };
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  /** \brief Calculate the LCP measure for registering src_cloud to target cloud
    * \param[in] T the transformation used for the registration
    */
  int calculateLCP(Eigen::Matrix4d &T);

  /** \brief Helper function for printTree */
  void printTreeHelper(MapNode* node);

  /********** Class Members ***********/
  /** \brief Root of the transformation tree*/
  MapNode *root_;

  /** \brief Max Depth of the transformation tree */
  int max_depth_;

  /** \brief  Dimensionality of the transformation Space. Rotations (3). Full Rigid(6) */
  int dim_;

  /** \brief  Best Transformation so far. Its LCP is saved in global_upper_bound_ */
  Eigen::Matrix4d bestT_;

  /** \brief LCP of best transformation so far */
  int global_upper_bound_;

  /** \brief The epsilon used in calculating the lcp upper bound */
  double epsilon_;

  /** \brief The source cloud that we are registering to the target cloud */
  std::vector<match_4pcs::Point3D> src_cloud_;

  /** \brief The search structure used to perform nn queries on the target cloud */
  Super4PCS::KdTree<double> tgt_search_;

  /** \brief  number of nodes created in the tree. used for debugging */
  int node_count_;

  /** \brief number of transformations that made it all the way to
     to the bottom of the tree and therefore their lcp
     was calculated */
  int num_lcps_;
};

#include "bnbTransformations.cpp"

#endif //#ifndef BNB_TRANSFORMATIONS
