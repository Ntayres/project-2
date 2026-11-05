# Process Reflection (1-2 pages)

Written by the student. Every answer must cite the transcript file (e.g., transcripts/05_2026-10-20_reviewer_step1.md) or commit hash it refers to.

## 1. What did the reviewer or tester catch that the builder missed? (at least two cases)

The reviewer had alot of small changes to reccomend, I believe this allowed for most tests to work perfectly. The only reason for some test failures were because the test had errors. Each review had atleast 10 changes to reccomend. For example step1_review found 10 including verifying integer values fit the chosen dtype before conversion. Step6_review had 11 reccomendation including per-year report validation not establishing complete model/year coverage.

## 2. One review comment you rejected, and why

There were some comments that were just made because of a bad prompt, I used a capital when addressing a file when the file was in all lower case confusing the reviwer. (R1 step1_review)

## 3. A prompt that did not work, and how you rewrote it

There were some times that the reviewer bot tried to give me a file just to state later that it could not actually make an md file. so I had to start asking the reveiwer to full print any info that would be in an review md file. (07_2026-11-2_reviewer-step2.md)

## 4. One other decision where you overrode an agent, and why

There were some changes that I just felt unneeded. The reviewer would flag how much of a priority things were, so i decided that some low priority suggestions werent neccesary. (R12 step2_review)

## 5. How you verified the agents' results (at least two results)

Running the test files by myself gave me things like step2_year_stats.cvs and step5_test_metrics.cvs. Running the same tests through copilot my tester gave the same or simular results. 

## 6. Parts you wrote or edited by hand

None
