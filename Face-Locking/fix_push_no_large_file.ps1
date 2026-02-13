# Fix push: remove large model from history by creating a fresh commit without it.
# Run from project root in PowerShell: .\fix_push_no_large_file.ps1

Set-Location $PSScriptRoot

# 1. Create orphan branch (no parent = no history with the large file)
git checkout --orphan main_new

# 2. Unstage everything, then add all EXCEPT models/ ( .gitignore has models/ and *.onnx )
git reset
git add -A
# Ensure models folder is not in index
git rm -r --cached models/ 2>$null
git add .gitignore
git add .

# 3. Commit current state (no large file)
git commit -m "ArcFace 5pt face recognition (model auto-downloaded on first run)"

# 4. Replace main with this clean history
git branch -D main
git branch -m main

# 5. Force push (overwrites remote main - no large file in history)
Write-Host "Ready. Run: git push --force origin main" -ForegroundColor Green
