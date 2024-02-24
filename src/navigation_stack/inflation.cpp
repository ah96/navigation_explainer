using costmap_2d::LETHAL_OBSTACLE = 254;
using costmap_2d::INSCRIBED_INFLATED_OBSTACLE = 253;
using costmap_2d::NO_INFORMATION;

inflation_radius by default is 0.55

inline unsigned char costLookup(int mx, int my, int src_x, int src_y)
{
  unsigned int dx = abs(mx - src_x);
  unsigned int dy = abs(my - src_y);
  return cached_costs_[dx][dy];
}

void InflationLayer::computeCaches()
{
  if (cell_inflation_radius_ == 0)
    return;

  // based on the inflation radius... compute distance and cost caches
  if (cell_inflation_radius_ != cached_cell_inflation_radius_)
  {
    deleteKernels();

    cached_costs_ = new unsigned char*[cell_inflation_radius_ + 2];
    cached_distances_ = new double*[cell_inflation_radius_ + 2];

    for (unsigned int i = 0; i <= cell_inflation_radius_ + 1; ++i)
    {
      cached_costs_[i] = new unsigned char[cell_inflation_radius_ + 2];
      cached_distances_[i] = new double[cell_inflation_radius_ + 2];
      for (unsigned int j = 0; j <= cell_inflation_radius_ + 1; ++j)
      {
        cached_distances_[i][j] = hypot(i, j);
      }
    }

    cached_cell_inflation_radius_ = cell_inflation_radius_;
  }

  for (unsigned int i = 0; i <= cell_inflation_radius_ + 1; ++i)
  {
    for (unsigned int j = 0; j <= cell_inflation_radius_ + 1; ++j)
    {
      cached_costs_[i][j] = computeCost(cached_distances_[i][j]);
    }
  }
}

void InflationLayer::setInflationParameters(double inflation_radius, double cost_scaling_factor)
{
  if (weight_ != cost_scaling_factor || inflation_radius_ != inflation_radius)
  {
    // Lock here so that reconfiguring the inflation radius doesn't cause segfaults
    // when accessing the cached arrays
    boost::unique_lock < boost::recursive_mutex > lock(*inflation_access_);

    inflation_radius_ = inflation_radius;
    cell_inflation_radius_ = cellDistance(inflation_radius_);
    weight_ = cost_scaling_factor;
    need_reinflation_ = true;
    computeCaches();
  }
}

