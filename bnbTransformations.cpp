#include "bnbTransformations.h"
using namespace std;
using namespace Eigen;
using namespace match_4pcs;

#define PI 3.1459

// TODO: make kdtree and src_cloud shared pointer members to avoid copying (faster).
template <unsigned char DIM>
bnbTransformations<DIM>::bnbTransformations(Super4PCS::KdTree<double> tgt_search,
                                            vector<Point3D> src_cloud, double epsilon)
: tgt_search_(tgt_search),
  src_cloud_(src_cloud),
  epsilon_(epsilon),
  node_count_(0),
  num_lcps_(0)
{
}

template <unsigned char DIM>
bnbTransformations<DIM>::~bnbTransformations()
{
  if (root_)
    delete root_;
  root_ = NULL;
}

//TODO; Implement Generic Init method
template <unsigned char DIM> void
bnbTransformations<DIM>::init (int tree_depth, double max_translation)
{
}

template <unsigned char DIM> void
bnbTransformations<DIM>::initForRotations(int max_depth)
{
  if (DIM != 3)
  {
    cerr << "Wrong dimensionality for rotation structure" << endl;
    //TODO: Replace exit with something more inline with RAII
    exit(1);
  }
  max_depth_ = max_depth;
  global_upper_bound_ = 0;
  bestT_ = Matrix4d::Zero();
  /* epsilon to prevent transformations falling outside the numerical limits of the octree which ca
     be caused by precision errors */
  double e = 0.001;
  double b[6] = {-PI-e,PI+e,
                      -PI-e,PI+e,
                      -PI-e,PI+e};
  vector<double> bounds(b, b+6);
  int lvl = 1;
  root_ = new MapNode(lvl, bounds);
  node_count_++;
  /* Best LCP is achieved when all points in src cloud are matched to target */
  root_->upper_bound_ = src_cloud_.size();
  cout << "root upper bound" << root_->upper_bound_ << endl;
}

template <unsigned char DIM> void
bnbTransformations<DIM>::add (Matrix4d &T)
{
  MapNode* current_node = root_;
  int current_upper = current_node->upper_bound_;
  while (global_upper_bound_ < current_upper && current_node->lvl_ < max_depth_)
  {
    AngleAxis<double> aa;
    aa.fromRotationMatrix(T.block<3,3>(0,0));
    int child_id = current_node->computeChildId(aa);
    current_node = current_node->getChild(*this,child_id);
    current_upper = current_node->upper_bound_;
  }

  
  /* When the global_upper_ bound is lower than the upper bound of the leaf containing the
   * transformation T, then T might achieve a better LCP than the
   * global upper bound.  Therefore, its LCP must calculated and the global upper bound
   * updated if need be.
   */
  if (global_upper_bound_ < current_upper && current_node->lvl_ == max_depth_)
  {
    /* update global upper bound if LCP(T) is higher */
    int lcp = calculateLCP(T);
    num_lcps_++;
    if (lcp > global_upper_bound_)
    {
      global_upper_bound_ = lcp;
      bestT_ = T;
    }
  }
}

template<unsigned char DIM> int
bnbTransformations<DIM>::calculateLCP(Matrix4d &T)
{
  int lcp = 0;
  double sqr_error;
  Eigen::Matrix<double,4,1> query_point;
  for (int i = 0; i < src_cloud_.size(); ++i)
  {
    match_4pcs::Point3D q = src_cloud_[i];

    query_point << q.x,
                   q.y,
                   q.z,
                    1;
    query_point = T * query_point;

    // calculate squared delta according to eqn 8
    sqr_error = pow(epsilon_,2);

    // perform the query
    typename Super4PCS::KdTree<double>::Index resId =
    tgt_search_.doQueryRestrictedClosest(query_point.head<3>(), sqr_error);
    if ( resId != Super4PCS::KdTree<double>::invalidIndex() ) {
       lcp++;
    }
   else
   {
      if (src_cloud_.size() - i + lcp < global_upper_bound_)
        break;
   }
  }
  return lcp;
}

template<unsigned char DIM> void
bnbTransformations<DIM>::printTree()
{
  if (root_ != NULL)
    printTreeHelper(root_);  
}

template <unsigned char DIM> Matrix4d
bnbTransformations<DIM>::getBestT()
{
  return bestT_;
}

template <unsigned char DIM> double
bnbTransformations<DIM>::getBestLCP()
{
  /* the LCP measure is the ratio of point in Q (src_cloud) that are matched to the target */
  return global_upper_bound_*1.0f/src_cloud_.size();
}

template <unsigned char DIM> void
bnbTransformations<DIM>::printTreeSize()
{
  cout << "Tree Size: " << node_count_ << endl;
}

template <unsigned char DIM> void
bnbTransformations<DIM>::printNumLCPS()
{
  cout << "Num LCPs Calculated: " << num_lcps_ << endl;
}
  
template<unsigned char DIM> void
bnbTransformations<DIM>::printTreeHelper (MapNode* node)
{
  cout << "(" << node->lvl_ << "," << node->children_.size() << ")";
  if (!node->children_.empty())
  {
    cout << "[";
    typename map<unsigned char, MapNode>::iterator it = node->children_.begin();
    for(; it != node->children_.end(); ++it)
      printTreeHelper(&(it->second));
    cout << "] ";
  }
}


