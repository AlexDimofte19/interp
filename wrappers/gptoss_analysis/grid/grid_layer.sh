# The gpt-oss P2 GRID layer: the one number every script of this line derives its layer and
# its paths from (gptoss_grid_mass_l{L}, heldout360_l{L}_grid, *_l{L}_* probes). Sourced, not run.
#
# It is the J-lens grid argmax of 1_loudest_layer/ (train 2,880, whole chain), under the
# GRID-ENVIRONMENT lens (/workspace/jlens/gridenv). 14 was that argmax under the wikitext lens
# in /workspace/jlens, which the first round used by mistake; rerun_gridenv.sh re-picks it
# and this default must be updated to what it prints before stage 2 runs.
GRID_LAYER=${GRID_LAYER:-14}
