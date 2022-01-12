CECI work follower.
Add .gitignore as a filter.

**Update Jan.4, 2022**
Extract some code from SubgraphMatching framework, it's stable with the commandline, which locate in sub_ceci folder.

**Usage**
Firstly, use `make` to build the project. Then, execute:
`./myceci -d test/sample_dataset/test_case_1.graph -q test/sample_dataset/query1_positive.graph -filter CECI -order CECI -engine CECI -num MAX`
Try also, 
`./myceci -d test/data_graph/HPRD.graph -q test/query_graph/query_dense_16_1.graph -filter CECI -order CECI -engine CECI -num MAX`

<TO DO LIST>
1. Study the functions of the CECI filter - order - exec process.
2. Try something to reconstruct the code.
3. As the input varies from CECI project and SubgraphMatching framework, Write a converter/change the input function of the code.

Other things
1. BTW, also find other datasets on SNAP, while the formats doesn't compatible with the CECI input. 
