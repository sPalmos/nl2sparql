# Onboard every ontology used in the paper. Run from anywhere:
#     powershell -File scripts\onboard_all.ps1
# Expects the graphs written by `python scripts\prepare_datasets.py build` (see README: Datasets):
# resources\knowledge_graphs\ and the whitespace-trimmed CK25 graph trimmed_sources\ck25.ttl.

Set-Location (Split-Path -Parent $PSScriptRoot)
$kg = "resources\knowledge_graphs"
$sp = "$kg\spider_correct"

$jobs = @(
  @{ f = "$kg\bestiary_sandro.rdf";              n = "bestiary" },
  @{ f = "trimmed_sources\ck25.ttl";             n = "ck25" },
  @{ f = "$sp\battle_death.rdf";                 n = "spider-battle-death" },
  @{ f = "$sp\car_1.rdf";                        n = "spider-car-1" },
  @{ f = "$sp\concert_singer.rdf";               n = "spider-concert-singer" },
  @{ f = "$sp\course_teach.rdf";                 n = "spider-course-teach" },
  @{ f = "$sp\cre_Doc_Template_Mgt.rdf";         n = "spider-cre-doc-template-mgt" },
  @{ f = "$sp\dog_kennels.rdf";                  n = "spider-dog-kennels" },
  @{ f = "$sp\employee_hire_evaluation.rdf";     n = "spider-employee-hire-evaluation" },
  @{ f = "$sp\flight_2.rdf";                     n = "spider-flight-2" },
  @{ f = "$sp\museum_visit.rdf";                 n = "spider-museum-visit" },
  @{ f = "$sp\network_1.rdf";                    n = "spider-network-1" },
  @{ f = "$sp\pets_1.rdf";                       n = "spider-pets-1" },
  @{ f = "$sp\poker_player.rdf";                 n = "spider-poker-player" },
  @{ f = "$sp\real_estate_properties.rdf";       n = "spider-real-estate-properties" },
  @{ f = "$sp\singer.rdf";                       n = "spider-singer" },
  @{ f = "$sp\student_transcripts_tracking.rdf"; n = "spider-student-transcripts-tracking" },
  @{ f = "$sp\tvshow.rdf";                       n = "spider-tvshow" },
  @{ f = "$sp\voter_1.rdf";                      n = "spider-voter-1" },
  @{ f = "$sp\world_1.rdf";                      n = "spider-world-1" }
)

$ok = @(); $failed = @()
foreach ($j in $jobs) {
  Write-Host "`n=== Onboarding $($j.n) ===" -ForegroundColor Cyan
  if (-not (Test-Path $j.f)) {
    Write-Host "MISSING FILE: $($j.f)" -ForegroundColor Red
    $failed += $j.n; continue
  }
  python onboard.py --ontology $j.f --output ontology_packages --name $j.n
  if ($LASTEXITCODE -eq 0) { $ok += $j.n } else { $failed += $j.n }
}

Write-Host "`n=== SUMMARY ===" -ForegroundColor Yellow
Write-Host "OK ($($ok.Count)): $($ok -join ', ')"
if ($failed.Count) { Write-Host "FAILED ($($failed.Count)): $($failed -join ', ')" -ForegroundColor Red }
