# GraphRAG
本库收录的是《GraphRAG实战》一书的开源资源，包括引用资料、图片、代码和样章等。<br><br>

本书用的是 [Neo4j Community Edition](https://we-yun.com/blog/prod-56.html) ，不支持多个图数据库同时在线，一次只能1个图数据库在线。所以，如果有多个应用有多个图，它们都存放在同一个图数据库中的话，它们的结点标签就不能相同，所使用的向量索引的名字也不能相同。<br><br>
第2章《微软 GraphRAG》中建立与使用的知识图谱，结点的标签是`__Document__`、`__Chunk__`、`__Entity__`与`__Community__`。<br><br>
第3章《Neo4j GraphRAG》中建立与使用的知识图谱，结点的标签是`Document`、`Chunk`、`Entity`、`__Community__`。<br><br>
第4章《开发GraphRAG应用》与第5章《Agent开发》建立与使用的知识图谱，结点的标签是`__Document__`、`__Chunk__`、`__Entity__`与`__Community__`，实体向量索引的名称是`vector`。<br><br>
第6章《在GraphRAG中应用国产大模型》，第7章《本地部署LLM》，第8章《开发GraphRAG APP》,第9章《GraphRAG 应用评估》建立与使用的知识图谱，结点的标签是`__Document2__`、`__Chunk2__`、`__Entity2__`与`__Community2__`，实体向量索引的名称是`vector2`。<br><br>
第2、3、4、5章建立和使用的3个知识图谱，它们的结点标签有重复，所以是不能同时存放在同一个图数据库中的，第6、7、8、9章更改了结点的标签与索引的名字，就可以放在一起。<br><br>
所以运行第3章代码的时候，要清空第2章建立的知识图谱，运行第4、5章代码的时候，要清空第3章建立的知识图谱，后面运行第6、7、8、9章代码时，就不需要清空第4、5章建立的知识图谱。<br><br>
清空Neo4j 图数据库，需要删除所有相关标签的结点和它们索引、约束，具体命令可参阅本库的`deletegraph.txt`。或者在Neo4j安装时第一次启动后就停止它，拷贝一个空的数据库，以后有需要时就通过拷贝并更名的方式切换当前数据库到一个空的数据库。<br><br>
```
(base) root@10-60-136-78:/opt/neo4j-chs-community-5.24.0-unix# cd ./data/databases
(base) root@10-60-136-78:/opt/neo4j-chs-community-5.24.0-unix/data/databases# ls
blank  demo  kgbuilder  msgraphrag  neo4j  store_lock  system
```
上面的图数据库中, blank就是在Neo4j第一次启动时，没有写入任何数据的情况下拷贝的空数据库。<br><br>
```
# cp -R neo4j blank
```
其它数据库demo、kgbuilder、msgraphrag等是拷贝当前有数据的数据库neo4j备份存档的，比如：<br><br>
```
# cp -R neo4j msgraphrag
```
以后可以从备份切换过来，比如：<br><br>
```
# rm -Rf neo4j
# cp -R msgraphrag neo4j
```
也可以修改Neo4j的配置文件指定使用不同的数据库来切换：<br><br>
```
(base) root@10-60-136-78:/opt/neo4j-chs-community-5.24.0-unix/conf# ls
neo4j-admin.conf  neo4j.conf  server-logs.xml  user-logs.xml
(base) root@10-60-136-78:/opt/neo4j-chs-community-5.24.0-unix/conf# vi neo4j.conf
```
修改启动时加载的数据库即可：<br><br>
```
# 启动时打开的数据库，社区版只能在线打开一个数据库，通过改变下面的名字切换。
initial.dbms.default_database=neo4j
# 切换时重建事务日志，否则不能启动，因为数据库变了，不匹配。
db.recovery.fail_on_missing_files=false
```