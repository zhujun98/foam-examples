#include <cassert>

#include <foamalgo/geometry_1m.hpp>
#include <foamalgo/imageproc.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>


int main()
{
  using namespace xt::placeholders;

  int n_pulses = 64;
  float nan = std::numeric_limits<float>::quiet_NaN();

  // ----------------------------------------------------------
  // assemble a train of LPD data (64 pulses/train, 16 modules)
  // ----------------------------------------------------------

  xt::xtensor<float, 4> src = xt::random::rand<float>({n_pulses, 16, 256, 256});
  xt::view(src, xt::all(), xt::all(), xt::all(), xt::range(_, _, 16)) = nan;

  auto geom = foam::LPD_1MGeometry(); // geometry to stack modules together

  xt::xtensor<float, 3> assembled = xt::empty<float>({n_pulses, 1024, 1024});
  geom.positionAllModules(src, assembled);

  // ------------------------------------------------------------
  // mask the assembled image by an image mask and threshold mask
  // ------------------------------------------------------------

  xt::xtensor<bool, 2> image_mask = xt::zeros<bool>({1024, 1024});
  xt::view(image_mask, xt::all(), xt::range(_, 16)) = true;

  foam::maskImageDataNan(assembled, image_mask, -1, 1);

  // -------------------------------------------------
  // calculate the nan-mean of images across the train
  // -------------------------------------------------

  auto assembled_mean = foam::nanmeanImageArray(assembled);

  return 0;
}
