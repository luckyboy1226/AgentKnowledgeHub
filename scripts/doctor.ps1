param([ValidateSet('core','full')][string]$Mode='core')

$ErrorActionPreference='Continue';$root=Split-Path $PSScriptRoot -Parent;$ok=$true
function Check($Name,$Action){try{if(& $Action){Write-Host "PASS $Name"}else{Write-Host "FAIL $Name";$script:ok=$false}}catch{Write-Host "FAIL $Name";$script:ok=$false}}
function Http200($Url){try{(Invoke-WebRequest $Url -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200}catch{$false}}
Check 'python/.env exists' {Test-Path "$root/python/.env"}
Check 'API health' {Http200 'http://127.0.0.1:8080/api/health'}
Check 'API readiness' {Http200 'http://127.0.0.1:8080/api/health/ready'}
Check 'frontend' {Http200 'http://127.0.0.1:8081/'}
Check 'frontend API proxy' {Http200 'http://127.0.0.1:8081/api/health'}
foreach($name in 'api','frontend'){$file=Join-Path $root ".runtime/$name.json";Check "$name PID state exists" {Test-Path $file};Check "$name listener matches PID state" {if(-not(Test-Path $file)){return $false};try{$state=Get-Content -Raw $file|ConvertFrom-Json;((Get-NetTCPConnection -LocalPort ([int]$state.port) -State Listen -EA SilentlyContinue|Select-Object -First 1).OwningProcess -eq $state.pid)}catch{$false}}}
Push-Location $root;try{Check 'core containers running' {$running=(docker compose ps --status running --services) -join ' ';$running -match 'mongodb' -and $running -match 'neo4j' -and $running -match 'chromadb'}}finally{Pop-Location}
if($Mode -eq 'full'){Write-Warning 'Kafka is optional infrastructure; CDC worker is NOT IMPLEMENTED and does not affect core readiness.'}
if(-not $ok){exit 1}
