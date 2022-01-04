
VertexID selectCECIStartVertex(const Graph *data_graph, const Graph *query_graph);
void generateCECIFilterPlan(const Graph *data_graph, const Graph *query_graph, TreeNode *&tree, VertexID *&order);
void bfsTraversal(const Graph *graph, VertexID root_vertex, TreeNode *&tree, VertexID *&bfs_order);
void allocateBuffer(const Graph *data_graph, const Graph *query_graph, ui **&candidates, ui *&candidates_count);
void computeCandidateWithNLF(const Graph *data_graph, const Graph *query_graph, VertexID query_vertex, ui &count, ui *buffer);
bool CECIFilter(const Graph *data_graph, const Graph *query_graph, ui **&candidates, ui *&candidates_count,
                           ui *&order, TreeNode *&tree,  std::vector<std::unordered_map<VertexID, std::vector<VertexID >>> &TE_Candidates,
                           std::vector<std::vector<std::unordered_map<VertexID, std::vector<VertexID>>>> &NTE_Candidates);
void computeCandidateWithNLF(const Graph *data_graph, const Graph *query_graph, VertexID query_vertex, ui &count, ui *buffer);
void generateCECIQueryPlan(const Graph *query_graph, TreeNode *tree, ui *bfs_order, ui *&order, ui *&pivot);
