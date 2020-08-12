1. 有序collect_set
```sql
select user_id, sort_array(collect_set(concat_ws(",", cast(pickup_datetime as string), cast(pickup_country_id AS string)))) 
from fact_order 
group by user_id 
limit 100a
```
