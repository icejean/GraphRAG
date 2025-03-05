# GraphRAG
<p>本库收录的是《GraphRAG实战》一书的开源资源，包括引用资料、图片、代码和样章等。</p>

<p>本书用的是[Neo4j Community Edition](https://we-yun.com/blog/prod-56.html)，不支持多个图数据库同时在线，一次只能1个图数据库在线。所以，如果有多个应用有多个图，它们都存放在同一个图数据库中的话，它们的结点标签就不能相同，所使用的向量索引的名字也不能相同。</p>
<p>第3章《Neo4j GraphRAG》中建立与使用的知识图谱，结点的标签是"Document"、"Chunk"、"Entity"、"__Community__"。</p>
<p>第4章《开发GraphRAG应用》与第5章《Agent开发》建立与使用的知识图谱，结点的标签是"__Document__"、"__Chunk__"、"__Entity__"与"__Community__"，实体向量索引的名称是"vector"。</p>
<p>第6章《在GraphRAG中应用国产大模型》，第7章《本地部署LLM》，第8章《开发GraphRAG APP》建立与使用的知识图谱，结点的标签是"__Document2__"、"__Chunk2__"、"__Entity2__"与"__Community2__"，实体向量索引的名称是"vector2"。</p>
<p>第2、3、4、5章建立和使用的3个知识图谱，它们的结点标签有重复，所以是不能同时存放在同一个图数据库中的，第6、7、8章更改了结点的标签与索引的名字，就可以放在一起。</p>
<p>所以运行第3章代码的时候，要清空第2章建立的知识图谱，运行第4、5章代码的时候，要清空第3章建立的知识图谱，后面运行第6、7、8、9章代码时，就不需要清空第4、5章建立的知识图谱。</p>
<p>清空Neo4j 图数据库，需要删除所有相关标签的结点和它们索引、约束，具体命令可参阅本库的deletegraph.txt。或者在Neo4j安装时第一次启动后就停止它，拷贝一个空的数据库，以后有需要时就通过拷贝并更名的方式切换当前数据库到一个空的数据库。</p>