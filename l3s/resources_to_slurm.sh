
## Example to move resources


PATH_RESOURCES_SOURCE=/data_ssds/disk02/ottowg/data/HGERE/saves
SSH_USER=slurm
PROJECT_PATH=/home/ottowg/projects/gsap/gsap-rel-related/hgere
PROJECT_PATH=/home/ottowg/projects/hgere
PATH_RESOURCES_TARGET=$SSH_USER:$PROJECT_PATH/saves
echo $PATH_RESOURCES_TARGET
for name in scier gsap-ere scinlp; do
  fn_source="$PATH_RESOURCES_SOURCE/$name/pre-filter/rules.json"
  fn_target="$PATH_RESOURCES_TARGET/$name/pre-filter/rules.json"
  rsync -av --mkpath --info=progress2 $fn_source $fn_target
  fn_source="$PATH_RESOURCES_SOURCE/$name/pruner/best"
  fn_target="$PATH_RESOURCES_TARGET/$name/pruner"
  rsync -av --mkpath --info=progress2 $fn_source $fn_target
done
