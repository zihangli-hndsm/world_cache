# Post-hoc shared-target train-source cluster

Shared validation target: `4fbad31e-465b-2a5d-84b7-c0ddea978db4`.
The 10 pairwise comparisons cover 5 train references.
All source transforms were estimated independently against the shared validation target.

| Train source A | Train source B | A→B / B→A coverage | NYU40 exact / marginal | Eigen13 exact / marginal | RIO27 exact / marginal |
|---|---|---:|---:|---:|---:|
| `0f2f271b-b736-2f71-8f3a-e9265463abcb` | `4a9a43e4-7736-2874-8479-0e81135ccdb0` | 9.5%/8.4% | 76.2%/17.6%; 74.9%/17.3% | 76.3%/22.1%; 75.4%/21.6% | 79.8%/19.1%; 79.5%/18.8% |
| `0f2f271b-b736-2f71-8f3a-e9265463abcb` | `4fbad31a-465b-2a5d-8566-f4e4845c1a78` | 11.9%/9.8% | 78.8%/15.8%; 77.1%/15.3% | 80.6%/18.6%; 78.6%/17.6% | 84.5%/17.0%; 83.3%/15.9% |
| `0f2f271b-b736-2f71-8f3a-e9265463abcb` | `5555108d-36f1-29c0-8b37-5efa2bef59d4` | 6.7%/8.3% | 74.8%/17.9%; 77.0%/18.2% | 79.7%/22.5%; 81.9%/23.0% | 83.9%/21.1%; 86.4%/20.4% |
| `4a9a43e4-7736-2874-8479-0e81135ccdb0` | `4fbad31a-465b-2a5d-8566-f4e4845c1a78` | 11.2%/10.3% | 77.4%/16.9%; 77.5%/16.5% | 74.9%/20.4%; 75.1%/19.8% | 79.2%/16.8%; 79.4%/16.4% |
| `4a9a43e4-7736-2874-8479-0e81135ccdb0` | `5555108d-36f1-29c0-8b37-5efa2bef59d4` | 9.2%/13.5% | 77.4%/19.5%; 80.4%/21.3% | 82.5%/24.0%; 84.2%/26.5% | 81.4%/20.7%; 84.7%/22.3% |
| `4fbad31a-465b-2a5d-8566-f4e4845c1a78` | `5555108d-36f1-29c0-8b37-5efa2bef59d4` | 7.5%/10.5% | 73.2%/16.9%; 74.9%/17.0% | 79.3%/20.8%; 79.6%/20.9% | 76.1%/18.1%; 78.5%/18.6% |
| `d7d40d5c-7a5d-2b36-9433-ee98b893988d` | `0f2f271b-b736-2f71-8f3a-e9265463abcb` | 9.6%/10.3% | 75.7%/14.9%; 79.2%/16.2% | 82.4%/17.6%; 85.5%/19.4% | 85.3%/15.7%; 86.6%/17.4% |
| `d7d40d5c-7a5d-2b36-9433-ee98b893988d` | `4a9a43e4-7736-2874-8479-0e81135ccdb0` | 13.3%/12.9% | 81.1%/16.9%; 80.0%/16.9% | 79.4%/20.6%; 78.4%/20.8% | 82.7%/16.9%; 81.7%/16.9% |
| `d7d40d5c-7a5d-2b36-9433-ee98b893988d` | `4fbad31a-465b-2a5d-8566-f4e4845c1a78` | 13.0%/11.7% | 82.3%/15.1%; 82.9%/14.8% | 83.0%/18.2%; 84.1%/17.4% | 83.7%/14.9%; 84.7%/14.6% |
| `d7d40d5c-7a5d-2b36-9433-ee98b893988d` | `5555108d-36f1-29c0-8b37-5efa2bef59d4` | 8.5%/11.5% | 77.0%/19.3%; 78.1%/19.5% | 82.3%/23.5%; 82.1%/23.9% | 79.9%/20.2%; 81.4%/20.3% |

Across directed class-field comparisons, exact agreement is 73.2%–86.6%; matched-subset marginal reference is 14.6%–26.5%; the lift ranges from 53.8% to 70.1%.

These source pairs were selected post hoc because they independently passed a geometric screen against the same validation reference. Matched-point outcomes are descriptive; spatial autocorrelation and selection are not accounted for, and this analysis does not prove that the scans depict the same physical site.

The marginal reference independently shuffles labels within the matched subsets. It is descriptive and is not a calibrated significance test.
