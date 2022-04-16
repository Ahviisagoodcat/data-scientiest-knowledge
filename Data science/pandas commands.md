# pandas command for data science

## change string format to datetime

```
#example "2017-01-01", "2017/1/1"
df[column]=pd.to_datetime(df[column])
```

## Group by
```
df.groupby([columns])[coluumn].sum()/min()/max()
```