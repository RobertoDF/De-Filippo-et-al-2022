import seaborn as sns
import pickle


with open("/temp/temp_summary_corrs.pkl", "rb") as fp:  # Unpickling
    summary_corrs = pickle.load(fp)

ax = sns.barplot(data=summary_corrs[summary_corrs["Count"]>2].sort_values(by="Count", ascending=False), x="Comparison", y="Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=75);
