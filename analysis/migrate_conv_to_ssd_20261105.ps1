$ErrorActionPreference = 'Stop'
$workspace = 'C:\Users\liang\Documents\Python Scripts\predify2021_my'
$destRoot = 'E:\Predify_Experiments_20261105'
$doc = Join-Path $workspace 'docs\conv_development_20261104'
$names = @('conv_development_20261104','conv_column_nonnegative_20261104')
New-Item -ItemType Directory -Force -Path $destRoot | Out-Null
function Status($stage,$name,$count) { @{stage=$stage;experiment=$name;verified_files=$count;updated=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $doc 'migration_status.json') }
foreach($name in $names) {
 $src = Join-Path $workspace ('Hebbian-CNN\Log\'+$name)
 $dst = Join-Path $destRoot $name
 $resolved = (Resolve-Path -LiteralPath $src).Path
 if($resolved -ne $src -or -not $resolved.StartsWith($workspace+'\Hebbian-CNN\Log\')) { throw 'Source boundary validation failed' }
 if((Get-Item -LiteralPath $src).Attributes -band [IO.FileAttributes]::ReparsePoint) {
  if((Get-Item -LiteralPath $src).Target -ne $dst -or -not (Test-Path -LiteralPath (Join-Path $doc ($name+'_migration_sha256.csv')))) { throw 'Unexpected source mapping' }
  Status 'already_mapped' $name 0
  continue
 }
 if([IO.Path]::GetFullPath($dst) -ne ($destRoot+'\'+$name)) { throw 'Destination boundary validation failed' }
 Status 'copying' $name 0
 & robocopy $src $dst /E /COPY:DAT /DCOPY:T /R:1 /W:1 /MT:8 /NP /NFL /NDL ("/LOG+:" + (Join-Path $doc 'migration_copy.log'))
 if($LASTEXITCODE -ge 8){throw "Copy failed: $LASTEXITCODE"}
 $files = @(Get-ChildItem -LiteralPath $src -File -Recurse)
 $audit = [System.Collections.Generic.List[object]]::new();$count=0
 foreach($file in $files) {
  $relative=$file.FullName.Substring($src.Length+1);$target=Join-Path $dst $relative
  if(-not (Test-Path -LiteralPath $target -PathType Leaf)){throw "Missing: $relative"}
  if((Get-Item -LiteralPath $target).Length -ne $file.Length){throw "Size mismatch: $relative"}
  $sh=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
  $dh=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
  if($sh -ne $dh){throw "Hash mismatch: $relative"}
  $audit.Add([PSCustomObject]@{relative=$relative;bytes=$file.Length;sha256=$sh});$count++
  if($count % 50 -eq 0){Status 'verifying' $name $count}
 }
 if(@(Get-ChildItem -LiteralPath $dst -File -Recurse).Count -ne $files.Count){throw 'Unexpected destination files'}
 $audit | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $doc ($name+'_migration_sha256.csv'))
 Status 'verified_removing_source' $name $count
 # Recursive removal is limited to the exact resolved experiment directory above;
 # every file has a matching size and SHA256 at the destination.
 if((Resolve-Path -LiteralPath $src).Path -ne $resolved){throw 'Source changed'}
 Remove-Item -LiteralPath $src -Recurse -Force
 New-Item -ItemType Junction -Path $src -Target $dst | Out-Null
 $link=Get-Item -LiteralPath $src
 if(-not ($link.Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'Junction creation failed'}
 Status 'mapped' $name $count
}
Status 'complete' 'both' 0

