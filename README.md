# Infer.NET-MACE

Infer.NET implementation of the MACE algorithm, as described in "Learning Whom to Trust With Mace" by Dirk Hovy et al, NAACL 2013.

The data is partially observed worker-item votes. The algorithm infers:
* true label posteriors
* spamming/trustworthiness parameter for each worker-item pair

The former can be used to gauge item difficulty (e.g. easy examples vs difficult ones that might need review and/or more votes). The latter can be used to detect workers of different skill, ranging from a complete spammer to a valuable worker.

## Data
The data must be formatted as a CSV file. Each line is a work item and column is a worker. A missing vote is marked with empty value. The header is required and is used to count the number of workers (worker names can be arbitrary). The category labels are starting from 0 and are expected to be continuous.

Sample data features votes from 8 workers on 10 work items. A work item label can be categorized in one of three categories. The data has been generated as:
- workers 1 through six make 1-2 mistakes in average
- worker 7 is a spammer; always provides label 0
- worker 8 is a perfect worker; always provides the correct answer
- each item gets 4 workers assigned (at random)

## How to read the output?
First block of the output is true label estimate for each work item. The second block informs about reliability/trust information for each worker-item pair. The latter output can be useful to find good/bad workers.

## References
Dirk Hovy, Taylor Berg-Kirkpatrick, Ashish Vaswani, Eduard Hovy, "Learning Whom to Trust with MACE", NAACL 2013.
http://www.aclweb.org/anthology/N13-1132

## License
Updated to use open source Infer.NET.
