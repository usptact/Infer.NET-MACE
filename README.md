# Infer.NET-MACE

Infer.NET implementation of the MACE algorithm, as described in "Learning Whom to Trust With Mace" by Dirk Hovy et al, NAACL 2013.

The data is partially observed worker-item votes. The algorithm infers:
* true label posterior
* spamming/trustworthiness parameter for each worker-item pair

The latter can be used to detect workers of different skill, ranging from a complete spammer to a valuable worker.

Paper:
http://www.aclweb.org/anthology/N13-1132
