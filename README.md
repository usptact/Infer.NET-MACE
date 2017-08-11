# Infer.NET-MACE

Infer.NET implementation of the MACE algorithm, as described in "Learning Whom to Trust With Mace" by Dirk Hovy et al, NAACL 2013.

The data is partially observed worker-item votes. The algorithm infers:
* true label posteriors
* spamming/trustworthiness parameter for each worker-item pair

The former can be used to gauge item difficulty (e.g. easy examples vs difficult ones that might need review and/or more votes). The latter can be used to detect workers of different skill, ranging from a complete spammer to a valuable worker.

## Data
The data must be formatted as a CSV file. Each line is a work item and column is a worker. A missing vote is marked with empty value. The header is required and is used to count the number of workers (worker names can be arbitrary).

## How to read the output?
First block of the output is true label estimate for each work item. The second block informs about reliability/trust information for each worker-item pair. The latter output can be useful to find good/bad workers.

## References
http://www.aclweb.org/anthology/N13-1132
