



```sql
SELECT 'cluster_A' as cluster,* FROM ml.Cluster_A
UNION 
SELECT 'cluster_B' as cluster,* FROM ml.Cluster_B
UNION 
SELECT 'cluster_C' as cluster,* FROM ml.Cluster_C
UNION 
SELECT 'cluster_D' as cluster,* FROM ml.Cluster_D
;
```

