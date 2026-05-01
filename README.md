# Fact-Checking Numerical Claims

The task spans across three languages - English, Spanish and Arabic.

As described on task website, Given the claim and a set of reasoning paths with evidence that was employed to generate those reasoning paths, your task is to rank the reasoning paths.
You can also verify which of your top-ranked paths has the right verdict.

During evaluation you would be asked to output top-5 reasoning paths as ranked by your verifier and we will evaluate Recall@5. Additionally we will derive top-1 verdict and calculate Macro F1.

The english train data is too big to fit in gitlab and can be found at : https://drive.google.com/file/d/11g-LyDVrMP09EimzKQIg0hXBvEXrKCQN/view?usp=sharing

The rest of the train and validation splits are in the folders in this repo.

Codabench competitions:

Arabic : https://www.codabench.org/competitions/15736
English: https://www.codabench.org/competitions/15572
Spanish: https://www.codabench.org/competitions/15718/