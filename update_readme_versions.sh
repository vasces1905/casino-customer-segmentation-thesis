# pulls last 10 commits with git log
  ## Finds the Version History heading
  ## Clears the old version history under it and adds a new one
  ## Does not touch other README content

#!/bin/bash
echo "## Version History" > version_tmp.md
git log --pretty=format:"- %h (%ad): %s" --date=short | head -n 10 >> version_tmp.md
awk '
BEGIN { version_block=0 }
/## Version History/ { version_block=1; print; while ((getline line < "'version_tmp.md'") > 0) print line; next }
version_block==1 && /^## / { version_block=0 }
version_block==0 { print }
' README.md > README_updated.md
mv README_updated.md README.md
rm version_tmp.md
echo "README.md version history updated!"
