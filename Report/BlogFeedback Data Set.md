# Analysing the 'BlogFeedback Data Set' from the UC Irvine Machine Learning repository

This notebook is used to analyze the 'BlogFeedback Data Set' from the UC Irvine Machine Learning repository. The data set is available [here](https://archive.ics.uci.edu/ml/datasets/BlogFeedback). **The objective of the notebook is to create a model to predict the number of comments in a blog post in the upcoming 24 hours**.

This data originates from blog posts. The raw HTML-documents of the blog posts were crawled and processed. In the train data, the basetimes were in the years 2010 and 2011. In the test data the basetimes were in February and March 2012.

**The data set has 280 attributes. Therefore, in this notebooks we test different techniques to deal with this large number of attributes**. First, we analyze the whole data set without any kind of adjustment. This will be our reference model. Then, we test some feature selection methods to identify the most relevant attributes to predict the target value. Finally, we test the Principal Component Analysis (PCA) dimensionality reduction method.

The notebook is organized as follows:

1. Data exploration
2. Train ML model
3. Evaluate the ML model
4. Conclusion

----------

## 1. Data exploration

In this section, we explore the characteristics of the data set, including its dimensions and characteristics of its variables.

The data set contains 281 columns and 52397 rows.

The attributes of the data set are the following:

Column:
- 1...50: Average, standard deviation, min, max and median of the Attributes 51...60 for the source of the current blog post. With source we mean the blog on which the post appeared. For example, myblog.blog.org would be the source of the post myblog.blog.org/post_2010_09_10
- 51: Total number of comments before basetime
- 52: Number of comments in the last 24 hours before the basetime
- 53: Let T1 denote the datetime 48 hours before basetime. Let T2 denote the datetime 24 hours before basetime. This attribute is the number of comments in the time period between T1 and T2
- 54: Number of comments in the first 24 hours after the publication of the blog post, but before basetime
- 55: The difference of Attribute 52 and Attribute 53
- 56...60: The same features as the attributes 51...55, but features 56...60 refer to the number of links (trackbacks), while features 51...55 refer to the number of comments.
- 61: The length of time between the publication of the blog post and basetime
- 62: The length of the blog post
- 63...262: The 200 bag of words features for 200 frequent words of the text of the blog post
- 263...269: binary indicator features (0 or 1) for the weekday (Monday...Sunday) of the basetime
- 270...276: binary indicator features (0 or 1) for the weekday (Monday...Sunday) of the date of publication of the blog post
- 277: Number of parent pages: we consider a blog post P as a parent of blog post B, if B is a reply (trackback) to blog post P.
- 278...280: Minimum, maximum, average number of comments that the parents received
- 281: The target: the number of comments in the next 24 hours (relative to basetime)


```python
import pandas as pd
import numpy as np
#!pip install -U scikit-learn
```

----------

### Getting the data


```python
attributes = [*range(1, 282, 1)]

df_data = pd.read_csv('/Users/leuzinger/Dropbox/Data Science/Awari/Regressions/BlogFeedback/blogData_train.csv',names=attributes)
df_data.reset_index(inplace=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 52397 entries, 0 to 52396
    Columns: 281 entries, 1 to 281
    dtypes: float64(281)
    memory usage: 112.3 MB



```python
att=[]
for i in ['total','last24h','24-48h','first24h','difference',
           'total_tr','last24h_tr','24-48h_tr','first24h_tr','difference_tr']:
    att1 = 'blog_avg_' + str(i)
    att2 = 'blog_std_' + str(i)
    att3 = 'blog_min_' + str(i)
    att4 = 'blog_max_' + str(i)
    att5 = 'blog_median_' + str(i)
    att.extend([att1,att2,att3,att4,att5])

att51_62 = ['total','last24h','24-48h','first24h','difference',
           'total_tr','last24h_tr','24-48h_tr','first24h_tr','difference_tr',
           'time_first_post','lenght_post']
att.extend(att51_62)

for i in range(63,263):
    att_word = 'word' + str(i-62)
    att.extend([att_word])

att263_281 = ['Mon_bl','Tue_bl','Wed_bl','Thu_bl','Fri_bl','Sat_bl','Sun_bl',
             'Mon_post','Tue_post','Wed_post','Thu_post','Fri_post','Sat_post','Sun_post',
             'parent_pages','min_parent','max_parent','avg_parent','target']
att.extend(att263_281)
```


```python
df_data.set_axis(att,axis=1,inplace=True)
df_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>blog_avg_total</th>
      <th>blog_std_total</th>
      <th>blog_min_total</th>
      <th>blog_max_total</th>
      <th>blog_median_total</th>
      <th>blog_avg_last24h</th>
      <th>blog_std_last24h</th>
      <th>blog_min_last24h</th>
      <th>blog_max_last24h</th>
      <th>blog_median_last24h</th>
      <th>blog_avg_24-48h</th>
      <th>blog_std_24-48h</th>
      <th>blog_min_24-48h</th>
      <th>blog_max_24-48h</th>
      <th>blog_median_24-48h</th>
      <th>blog_avg_first24h</th>
      <th>blog_std_first24h</th>
      <th>blog_min_first24h</th>
      <th>blog_max_first24h</th>
      <th>blog_median_first24h</th>
      <th>blog_avg_difference</th>
      <th>blog_std_difference</th>
      <th>blog_min_difference</th>
      <th>blog_max_difference</th>
      <th>blog_median_difference</th>
      <th>blog_avg_total_tr</th>
      <th>blog_std_total_tr</th>
      <th>blog_min_total_tr</th>
      <th>blog_max_total_tr</th>
      <th>blog_median_total_tr</th>
      <th>blog_avg_last24h_tr</th>
      <th>blog_std_last24h_tr</th>
      <th>blog_min_last24h_tr</th>
      <th>blog_max_last24h_tr</th>
      <th>blog_median_last24h_tr</th>
      <th>blog_avg_24-48h_tr</th>
      <th>blog_std_24-48h_tr</th>
      <th>blog_min_24-48h_tr</th>
      <th>blog_max_24-48h_tr</th>
      <th>blog_median_24-48h_tr</th>
      <th>blog_avg_first24h_tr</th>
      <th>blog_std_first24h_tr</th>
      <th>blog_min_first24h_tr</th>
      <th>blog_max_first24h_tr</th>
      <th>blog_median_first24h_tr</th>
      <th>blog_avg_difference_tr</th>
      <th>blog_std_difference_tr</th>
      <th>blog_min_difference_tr</th>
      <th>blog_max_difference_tr</th>
      <th>blog_median_difference_tr</th>
      <th>total</th>
      <th>last24h</th>
      <th>24-48h</th>
      <th>first24h</th>
      <th>difference</th>
      <th>total_tr</th>
      <th>last24h_tr</th>
      <th>24-48h_tr</th>
      <th>first24h_tr</th>
      <th>difference_tr</th>
      <th>time_first_post</th>
      <th>lenght_post</th>
      <th>word1</th>
      <th>word2</th>
      <th>word3</th>
      <th>word4</th>
      <th>word5</th>
      <th>word6</th>
      <th>word7</th>
      <th>word8</th>
      <th>word9</th>
      <th>word10</th>
      <th>word11</th>
      <th>word12</th>
      <th>word13</th>
      <th>word14</th>
      <th>word15</th>
      <th>word16</th>
      <th>word17</th>
      <th>word18</th>
      <th>word19</th>
      <th>word20</th>
      <th>word21</th>
      <th>word22</th>
      <th>word23</th>
      <th>word24</th>
      <th>word25</th>
      <th>word26</th>
      <th>word27</th>
      <th>word28</th>
      <th>word29</th>
      <th>word30</th>
      <th>word31</th>
      <th>word32</th>
      <th>word33</th>
      <th>word34</th>
      <th>word35</th>
      <th>word36</th>
      <th>word37</th>
      <th>word38</th>
      <th>word39</th>
      <th>word40</th>
      <th>word41</th>
      <th>word42</th>
      <th>word43</th>
      <th>word44</th>
      <th>word45</th>
      <th>word46</th>
      <th>word47</th>
      <th>word48</th>
      <th>word49</th>
      <th>word50</th>
      <th>word51</th>
      <th>word52</th>
      <th>word53</th>
      <th>word54</th>
      <th>word55</th>
      <th>word56</th>
      <th>word57</th>
      <th>word58</th>
      <th>word59</th>
      <th>word60</th>
      <th>word61</th>
      <th>word62</th>
      <th>word63</th>
      <th>word64</th>
      <th>word65</th>
      <th>word66</th>
      <th>word67</th>
      <th>word68</th>
      <th>word69</th>
      <th>word70</th>
      <th>word71</th>
      <th>word72</th>
      <th>word73</th>
      <th>word74</th>
      <th>word75</th>
      <th>word76</th>
      <th>word77</th>
      <th>word78</th>
      <th>word79</th>
      <th>word80</th>
      <th>word81</th>
      <th>word82</th>
      <th>word83</th>
      <th>word84</th>
      <th>word85</th>
      <th>word86</th>
      <th>word87</th>
      <th>word88</th>
      <th>word89</th>
      <th>word90</th>
      <th>word91</th>
      <th>word92</th>
      <th>word93</th>
      <th>word94</th>
      <th>word95</th>
      <th>word96</th>
      <th>word97</th>
      <th>word98</th>
      <th>word99</th>
      <th>word100</th>
      <th>word101</th>
      <th>word102</th>
      <th>word103</th>
      <th>word104</th>
      <th>word105</th>
      <th>word106</th>
      <th>word107</th>
      <th>word108</th>
      <th>word109</th>
      <th>word110</th>
      <th>word111</th>
      <th>word112</th>
      <th>word113</th>
      <th>word114</th>
      <th>word115</th>
      <th>word116</th>
      <th>word117</th>
      <th>word118</th>
      <th>word119</th>
      <th>word120</th>
      <th>word121</th>
      <th>word122</th>
      <th>word123</th>
      <th>word124</th>
      <th>word125</th>
      <th>word126</th>
      <th>word127</th>
      <th>word128</th>
      <th>word129</th>
      <th>word130</th>
      <th>word131</th>
      <th>word132</th>
      <th>word133</th>
      <th>word134</th>
      <th>word135</th>
      <th>word136</th>
      <th>word137</th>
      <th>word138</th>
      <th>word139</th>
      <th>word140</th>
      <th>word141</th>
      <th>word142</th>
      <th>word143</th>
      <th>word144</th>
      <th>word145</th>
      <th>word146</th>
      <th>word147</th>
      <th>word148</th>
      <th>word149</th>
      <th>word150</th>
      <th>word151</th>
      <th>word152</th>
      <th>word153</th>
      <th>word154</th>
      <th>word155</th>
      <th>word156</th>
      <th>word157</th>
      <th>word158</th>
      <th>word159</th>
      <th>word160</th>
      <th>word161</th>
      <th>word162</th>
      <th>word163</th>
      <th>word164</th>
      <th>word165</th>
      <th>word166</th>
      <th>word167</th>
      <th>word168</th>
      <th>word169</th>
      <th>word170</th>
      <th>word171</th>
      <th>word172</th>
      <th>word173</th>
      <th>word174</th>
      <th>word175</th>
      <th>word176</th>
      <th>word177</th>
      <th>word178</th>
      <th>word179</th>
      <th>word180</th>
      <th>word181</th>
      <th>word182</th>
      <th>word183</th>
      <th>word184</th>
      <th>word185</th>
      <th>word186</th>
      <th>word187</th>
      <th>word188</th>
      <th>word189</th>
      <th>word190</th>
      <th>word191</th>
      <th>word192</th>
      <th>word193</th>
      <th>word194</th>
      <th>word195</th>
      <th>word196</th>
      <th>word197</th>
      <th>word198</th>
      <th>word199</th>
      <th>word200</th>
      <th>Mon_bl</th>
      <th>Tue_bl</th>
      <th>Wed_bl</th>
      <th>Thu_bl</th>
      <th>Fri_bl</th>
      <th>Sat_bl</th>
      <th>Sun_bl</th>
      <th>Mon_post</th>
      <th>Tue_post</th>
      <th>Wed_post</th>
      <th>Thu_post</th>
      <th>Fri_post</th>
      <th>Sat_post</th>
      <th>Sun_post</th>
      <th>parent_pages</th>
      <th>min_parent</th>
      <th>max_parent</th>
      <th>avg_parent</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40.30467</td>
      <td>53.845657</td>
      <td>0.0</td>
      <td>401.0</td>
      <td>15.0</td>
      <td>15.52416</td>
      <td>32.44188</td>
      <td>0.0</td>
      <td>377.0</td>
      <td>3.0</td>
      <td>14.044226</td>
      <td>32.615417</td>
      <td>0.0</td>
      <td>377.0</td>
      <td>2.0</td>
      <td>34.567566</td>
      <td>48.475178</td>
      <td>0.0</td>
      <td>378.0</td>
      <td>12.0</td>
      <td>1.479934</td>
      <td>46.18691</td>
      <td>-356.0</td>
      <td>377.0</td>
      <td>0.0</td>
      <td>1.076167</td>
      <td>1.795416</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.400491</td>
      <td>1.078097</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.377559</td>
      <td>1.07421</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.972973</td>
      <td>1.704671</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.022932</td>
      <td>1.521174</td>
      <td>-8.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40.30467</td>
      <td>53.845657</td>
      <td>0.0</td>
      <td>401.0</td>
      <td>15.0</td>
      <td>15.52416</td>
      <td>32.44188</td>
      <td>0.0</td>
      <td>377.0</td>
      <td>3.0</td>
      <td>14.044226</td>
      <td>32.615417</td>
      <td>0.0</td>
      <td>377.0</td>
      <td>2.0</td>
      <td>34.567566</td>
      <td>48.475178</td>
      <td>0.0</td>
      <td>378.0</td>
      <td>12.0</td>
      <td>1.479934</td>
      <td>46.18691</td>
      <td>-356.0</td>
      <td>377.0</td>
      <td>0.0</td>
      <td>1.076167</td>
      <td>1.795416</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.400491</td>
      <td>1.078097</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.377559</td>
      <td>1.07421</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.972973</td>
      <td>1.704671</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.022932</td>
      <td>1.521174</td>
      <td>-8.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>-2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40.30467</td>
      <td>53.845657</td>
      <td>0.0</td>
      <td>401.0</td>
      <td>15.0</td>
      <td>15.52416</td>
      <td>32.44188</td>
      <td>0.0</td>
      <td>377.0</td>
      <td>3.0</td>
      <td>14.044226</td>
      <td>32.615417</td>
      <td>0.0</td>
      <td>377.0</td>
      <td>2.0</td>
      <td>34.567566</td>
      <td>48.475178</td>
      <td>0.0</td>
      <td>378.0</td>
      <td>12.0</td>
      <td>1.479934</td>
      <td>46.18691</td>
      <td>-356.0</td>
      <td>377.0</td>
      <td>0.0</td>
      <td>1.076167</td>
      <td>1.795416</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.400491</td>
      <td>1.078097</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.377559</td>
      <td>1.07421</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.972973</td>
      <td>1.704671</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.022932</td>
      <td>1.521174</td>
      <td>-8.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>-2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40.30467</td>
      <td>53.845657</td>
      <td>0.0</td>
      <td>401.0</td>
      <td>15.0</td>
      <td>15.52416</td>
      <td>32.44188</td>
      <td>0.0</td>
      <td>377.0</td>
      <td>3.0</td>
      <td>14.044226</td>
      <td>32.615417</td>
      <td>0.0</td>
      <td>377.0</td>
      <td>2.0</td>
      <td>34.567566</td>
      <td>48.475178</td>
      <td>0.0</td>
      <td>378.0</td>
      <td>12.0</td>
      <td>1.479934</td>
      <td>46.18691</td>
      <td>-356.0</td>
      <td>377.0</td>
      <td>0.0</td>
      <td>1.076167</td>
      <td>1.795416</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.400491</td>
      <td>1.078097</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.377559</td>
      <td>1.07421</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.972973</td>
      <td>1.704671</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.022932</td>
      <td>1.521174</td>
      <td>-8.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40.30467</td>
      <td>53.845657</td>
      <td>0.0</td>
      <td>401.0</td>
      <td>15.0</td>
      <td>15.52416</td>
      <td>32.44188</td>
      <td>0.0</td>
      <td>377.0</td>
      <td>3.0</td>
      <td>14.044226</td>
      <td>32.615417</td>
      <td>0.0</td>
      <td>377.0</td>
      <td>2.0</td>
      <td>34.567566</td>
      <td>48.475178</td>
      <td>0.0</td>
      <td>378.0</td>
      <td>12.0</td>
      <td>1.479934</td>
      <td>46.18691</td>
      <td>-356.0</td>
      <td>377.0</td>
      <td>0.0</td>
      <td>1.076167</td>
      <td>1.795416</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>0.400491</td>
      <td>1.078097</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.377559</td>
      <td>1.07421</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.972973</td>
      <td>1.704671</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.022932</td>
      <td>1.521174</td>
      <td>-8.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>



----------

### Data correlation

We start our analysis looking to which attributes have the higher correlation with the price. First, we create a correlation matrix. 

We can see that the varibales that have the stronger postive correlations with the target value are the blog_median_last24h and blog_avg_difference. Besides, we see that the total blog publications, publications in the last 24h and publications between 24h-48h have a strong correlation with each other.


```python
corr_matrix = df_data.drop(df_data.iloc[:,261:277],axis=1).corr()
corr_matrix.loc['target'].sort_values(ascending=False)
```




    target                       1.000000
    blog_median_last24h          0.506540
    blog_avg_difference          0.503375
    blog_avg_last24h             0.497631
    blog_median_total            0.491707
    blog_avg_24-48h              0.490111
    blog_median_24-48h           0.489674
    blog_median_first24h         0.486316
    blog_avg_total               0.485464
    last24h                      0.472061
    blog_avg_first24h            0.471999
    blog_median_last24h_tr       0.461627
    blog_std_difference          0.440003
    blog_std_24-48h              0.439152
    blog_std_last24h             0.433578
    blog_std_total               0.424616
    blog_std_first24h            0.384654
    blog_max_total               0.356604
    blog_median_total_tr         0.338961
    blog_avg_24-48h_tr           0.337775
    blog_avg_last24h_tr          0.335829
    blog_avg_first24h_tr         0.329670
    blog_avg_total_tr            0.328525
    blog_median_first24h_tr      0.323661
    blog_max_24-48h              0.322775
    blog_max_last24h             0.322106
    blog_max_difference          0.320133
    total                        0.314446
    first24h                     0.314177
    blog_max_first24h            0.299688
    difference                   0.296273
    blog_std_difference_tr       0.292805
    blog_std_24-48h_tr           0.285755
    blog_std_last24h_tr          0.283884
    blog_std_total_tr            0.266815
    blog_std_first24h_tr         0.265203
    last24h_tr                   0.260903
    blog_max_last24h_tr          0.251493
    blog_max_24-48h_tr           0.251485
    blog_max_total_tr            0.247457
    blog_max_difference_tr       0.245544
    blog_avg_difference_tr       0.233080
    blog_max_first24h_tr         0.232089
    first24h_tr                  0.198638
    total_tr                     0.191917
    difference_tr                0.146145
    24-48h                       0.117642
    word92                       0.080473
    24-48h_tr                    0.067141
    word184                      0.064753
    word5                        0.064112
    word96                       0.063923
    word170                      0.063903
    word39                       0.061460
    word7                        0.061238
    word81                       0.060322
    word148                      0.058703
    word186                      0.055606
    word132                      0.055318
    word77                       0.054750
    blog_min_first24h            0.053221
    blog_min_total               0.053221
    word164                      0.051995
    word122                      0.051541
    word40                       0.050907
    word151                      0.049974
    lenght_post                  0.048209
    word129                      0.045419
    word63                       0.045107
    word17                       0.044715
    word118                      0.043835
    word89                       0.043425
    word171                      0.041580
    word157                      0.040974
    word133                      0.038016
    word58                       0.037928
    word131                      0.035357
    word97                       0.035195
    blog_min_last24h             0.034916
    word140                      0.034695
    word2                        0.033752
    word15                       0.033374
    word159                      0.033300
    word60                       0.032340
    word10                       0.031312
    word145                      0.030371
    word179                      0.027677
    word34                       0.027294
    word135                      0.025934
    word53                       0.025293
    word52                       0.025017
    word168                      0.024109
    word121                      0.024025
    word75                       0.023639
    word165                      0.022007
    word6                        0.021817
    word185                      0.021600
    word74                       0.021484
    word79                       0.020781
    word72                       0.019945
    word134                      0.019839
    word199                      0.019466
    word42                       0.018614
    word109                      0.018127
    word76                       0.017458
    word190                      0.017291
    word172                      0.017250
    word73                       0.017025
    word66                       0.016475
    word144                      0.016079
    word26                       0.016067
    word166                      0.015934
    word101                      0.015695
    word156                      0.015629
    word78                       0.015366
    word146                      0.015329
    word54                       0.013581
    word143                      0.013384
    word195                      0.013151
    word192                      0.013099
    word23                       0.012943
    word154                      0.012197
    word124                      0.010315
    word30                       0.010222
    word119                      0.010158
    word41                       0.009785
    word112                      0.009739
    word27                       0.009347
    word193                      0.008334
    word113                      0.008176
    word100                      0.007944
    word48                       0.007736
    word127                      0.007250
    word36                       0.007120
    word153                      0.007086
    word147                      0.007064
    word56                       0.006732
    word114                      0.006725
    word57                       0.006716
    word125                      0.006525
    word43                       0.006345
    word130                      0.005843
    word102                      0.005558
    word198                      0.005246
    word91                       0.005214
    word13                       0.005094
    word88                       0.004966
    word64                       0.004865
    word152                      0.004798
    word65                       0.004748
    word1                        0.004429
    word163                      0.004345
    word46                       0.004174
    word197                      0.003610
    word183                      0.003577
    word55                       0.003400
    word38                       0.003389
    word108                      0.003139
    word71                       0.002858
    word28                       0.002787
    word20                       0.002537
    blog_median_difference_tr    0.002513
    word98                       0.002406
    blog_median_24-48h_tr        0.002224
    word62                       0.002174
    word59                       0.002005
    word37                       0.001608
    word21                       0.001515
    word196                      0.001495
    word8                        0.001491
    word149                      0.001472
    word51                       0.000949
    word137                      0.000862
    word175                      0.000618
    word12                       0.000457
    word169                      0.000237
    word120                      0.000228
    word80                       0.000096
    word176                     -0.000001
    word155                     -0.000077
    word29                      -0.000181
    word4                       -0.000268
    word178                     -0.000353
    word24                      -0.000449
    word160                     -0.000473
    word31                      -0.000574
    word187                     -0.000735
    word177                     -0.000786
    word44                      -0.000896
    word162                     -0.001137
    blog_min_first24h_tr        -0.001228
    blog_min_total_tr           -0.001228
    word194                     -0.001284
    avg_parent                  -0.001354
    word115                     -0.001527
    word18                      -0.001568
    word19                      -0.001568
    word150                     -0.001568
    word14                      -0.001568
    word188                     -0.001568
    word11                      -0.001568
    word181                     -0.001568
    word174                     -0.001568
    word128                     -0.001568
    word99                      -0.001568
    word35                      -0.001568
    word3                       -0.001568
    word94                      -0.001568
    word107                     -0.001568
    word110                     -0.001568
    word87                      -0.001568
    word32                      -0.001568
    word117                     -0.001568
    word104                     -0.001568
    word70                      -0.001568
    word136                     -0.001568
    word138                     -0.001568
    word68                      -0.001568
    word123                     -0.001769
    word61                      -0.001802
    word67                      -0.001811
    word33                      -0.001929
    word22                      -0.001966
    word49                      -0.001966
    word182                     -0.001986
    word103                     -0.001993
    word161                     -0.002030
    word93                      -0.002030
    word47                      -0.002074
    word106                     -0.002221
    word25                      -0.002259
    max_parent                  -0.002362
    word191                     -0.002369
    word86                      -0.002408
    word105                     -0.002411
    word95                      -0.002447
    word111                     -0.002469
    word167                     -0.002660
    word173                     -0.002725
    word180                     -0.002754
    word50                      -0.002985
    word141                     -0.003541
    word16                      -0.003553
    word116                     -0.003677
    word69                      -0.003723
    word45                      -0.003724
    word142                     -0.004113
    blog_median_difference      -0.004137
    word85                      -0.004241
    word82                      -0.004510
    word139                     -0.004516
    word126                     -0.005383
    word90                      -0.005481
    word189                     -0.005553
    word84                      -0.005726
    word9                       -0.005750
    word83                      -0.006695
    word158                     -0.006848
    time_first_post             -0.152908
    blog_min_difference_tr      -0.230493
    blog_min_difference         -0.280792
    blog_min_24-48h                   NaN
    blog_min_last24h_tr               NaN
    blog_min_24-48h_tr                NaN
    min_parent                        NaN
    Name: target, dtype: float64



----------

### Creating the Train and Test sets

Creating a test set at the beginning of the project avoid *data snooping* bias, i.e., "when you estimate the generalization error using the test set, your estimate will be too optimistic, and you will launch a system that will not perform as well as expected" (GÉRON, 2019).

In this data set, the test set has already been divided. Therefore, we do not need to create a test set, just separete the target value from the other attributes to create our training set.


```python
blog_X_train = df_data.drop('target',axis=1).copy()
blog_y_train = df_data['target'].copy()
```

----------

### Preparing the data for ML algorithms

Before creating the ML models, we need to prepare the data so that the ML algorithms will work properly.

First, we need to clean missing values from the dataset. Second, we need to put all the attributes in the same scale because "Machine Learning algorithms don’t perform well when the input numerical attributes have very different scales" [(GÉRON, 2019)](https://www.amazon.com.br/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646).

We verify that there is no missing values in our data set. So, we just prepare a pipeline to do the scaling when necessary.


```python
blog_X_train.isnull().values.any()
```




    False




```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def estimator_transf(estimator):
    pipeline = Pipeline(steps=[('m', estimator)])
    return pipeline

def estimator_scaler(estimator):
    pipeline = Pipeline(steps=[('scaler',StandardScaler()),('model', estimator)])
    return pipeline 
```

----------

## 2. Train ML model

After preparing the data set, we are ready to select and train our ML model.

We start with a Linear Regression (LR) model. "A regression model, such as linear regression, models an output value based on a linear combination of input values" [(BROWNLEE, 2020)](https://machinelearningmastery.com/introduction-to-time-series-forecasting-with-python/).

Then, we try some regularized linear models. This kind of model constrain the weights of the model, avoiding overfitting (GÉRON, 2019). We try three regularized linear models [(BROWNLEE, 2016)](https://machinelearningmastery.com/machine-learning-with-python/):

1. Ridge regression. This model model uses the L2 regularization. It adds “squared magnitude” of coefficient as a penalty term to the loss function [(NAGPAL, 2017)](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c).
2. Lasso regression. This model model uses the L1 regularization. It adds “absolute value of magnitude” of coefficient as penalty term to the loss function (NAGPAL, 2017).
3. Elastic Net. This model combines the Ridge and the Lasso models. "It seeks to minimize the complexity of the regression model (magnitude and number of regression coefficients) by penalizing the model using both the L2-norm (sum squared coefficient values) and the L1-norm (sum absolute coefficient values)" (BROWNLEE, 2016).

Finally, we also try some nonlinear algorithms:

1. Classification and Regression Trees (CART). It uses "the train- ing data to select the best points to split the data in order to minimize a cost metric" (BROWNLEE, 2016).
2. k-Nearest Neighbors (KNN). This model "locates the k most similar instances in the training dataset for a new data instance" (BROWNLEE, 2016).

The models are evaluated using the mean absolute error (MAE), root square mean error (RMSE), and R². RMSE punish larger errors more than smaller errors, inflating or magnifying the mean error score. This is due to the square of the error value. MAE does not give more or less weight to different types of errors and instead the scores increase linearly with increases in error. MAE is the simplest evaluation metric and most easily interpreted. R² tells you how much variance your model accounts for. In the case of the MAE and RMSE, the lower the better. But for R², the close the value is to 1, the better ([HALE, 2020](https://towardsdatascience.com/which-evaluation-metric-should-you-use-in-machine-learning-regression-problems-20cdaef258e); [BROWNLEE, 2021](https://machinelearningmastery.com/regression-metrics-for-machine-learning/)).

Besides, "the key to a fair comparison of machine learning algorithms is ensuring that each algorithm is evaluated in the same way on the same data. You can achieve this by forcing each algorithm to be evaluated on a consistent test harness" (BROWNLEE, 2016). In this project, we do this by using the same split in the cross validation. We use the KFold function from the sklearn library with a random value rs as the random_state parameter. Although the rs value change everytime the notebook is run, once it is set, the same rs value is used in all the models. This guarantees that all the models are evaluated on the same data.

The result of the tests of the models with the training data shows that **the KNN is the best model**. It has the lowest MAE and RMSE, and the highest R².

However, differing scales of the raw data could be negatively impacting the performance of some of the models. Therefore, we test the models again, but this time we standardize the data set.

We can see that the performance of most models improved with standardization. However, the performance of the KNN degraded with the standardized data. Even so, KNN was still the best method.

**Therefore, for this initial test, we verify that KNN without standardization is the best model for our data**.

However, **using the data set with all the 280 attributes requires a lot of computing time**. So, let's try some featuring selection methods to see if we can reduce the number of attributes to be used in our models.


```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

def estimator_cross_val (model,estimator,pipe,matriz,rs,X,y):
    pipe_ = pipe(estimator)
    scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error','r2']
    kfold = KFold(n_splits=5, random_state=rs,shuffle=True)
    scores = cross_validate(pipe_,X,y,cv=kfold,scoring=scoring)
    
    mae_scores = -scores.get('test_neg_mean_absolute_error')
    mae_mean = mae_scores.mean()
    mae_std = mae_scores.std()
    
    rmse_scores = -scores.get('test_neg_root_mean_squared_error')
    rmse_mean = rmse_scores.mean()
    rmse_std = rmse_scores.std()
    
    r2_scores = scores.get('test_r2')
    r2_mean = r2_scores.mean()
    r2_std = r2_scores.std()
    
    results_ = [model,mae_mean,mae_std,rmse_mean,rmse_std,r2_mean,r2_std]
    results_ = pd.Series(results_, index = matriz.columns)
    results = matriz.append(results_,ignore_index=True)
    return results
```


```python
from random import randrange
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import warnings

warnings.filterwarnings("ignore")

rs = randrange(10000)
matriz = pd.DataFrame(columns=['model','MAE_mean','MAE_std','RMSE_mean','RMSE_std','R2_mean','R2_std'])

matriz = estimator_cross_val('Linear Regression',LinearRegression(),estimator_transf,matriz,rs,blog_X_train,blog_y_train)
matriz = estimator_cross_val('Ridge Regression',Ridge(),estimator_transf,matriz,rs,blog_X_train,blog_y_train)
matriz = estimator_cross_val('Lasso',Lasso(),estimator_transf,matriz,rs,blog_X_train,blog_y_train)
matriz = estimator_cross_val('Elastic Net',ElasticNet(),estimator_transf,matriz,rs,blog_X_train,blog_y_train)
matriz = estimator_cross_val('KNN',KNeighborsRegressor(),estimator_transf,matriz,rs,blog_X_train,blog_y_train)
matriz = estimator_cross_val('CART',DecisionTreeRegressor(),estimator_transf,matriz,rs,blog_X_train,blog_y_train)
matriz
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>MAE_mean</th>
      <th>MAE_std</th>
      <th>RMSE_mean</th>
      <th>RMSE_std</th>
      <th>R2_mean</th>
      <th>R2_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>9.535883</td>
      <td>0.155004</td>
      <td>30.383434</td>
      <td>1.101252</td>
      <td>0.347938</td>
      <td>0.030813</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ridge Regression</td>
      <td>9.530001</td>
      <td>0.155896</td>
      <td>30.378057</td>
      <td>1.104894</td>
      <td>0.348174</td>
      <td>0.030823</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lasso</td>
      <td>9.122793</td>
      <td>0.183914</td>
      <td>30.302539</td>
      <td>1.137340</td>
      <td>0.351477</td>
      <td>0.030541</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Elastic Net</td>
      <td>9.135748</td>
      <td>0.185702</td>
      <td>30.309116</td>
      <td>1.139229</td>
      <td>0.351195</td>
      <td>0.030631</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNN</td>
      <td>6.387953</td>
      <td>0.228165</td>
      <td>28.589911</td>
      <td>1.318047</td>
      <td>0.423054</td>
      <td>0.028169</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CART</td>
      <td>6.373964</td>
      <td>0.214273</td>
      <td>33.118618</td>
      <td>1.396891</td>
      <td>0.220945</td>
      <td>0.091865</td>
    </tr>
  </tbody>
</table>
</div>




```python
matriz2 = pd.DataFrame(columns=['model','MAE_mean','MAE_std','RMSE_mean','RMSE_std','R2_mean','R2_std'])

matriz2 = estimator_cross_val('Linear Regression',LinearRegression(),estimator_scaler,matriz2,rs,blog_X_train,blog_y_train)
matriz2 = estimator_cross_val('Ridge Regression',Ridge(),estimator_scaler,matriz2,rs,blog_X_train,blog_y_train)
matriz2 = estimator_cross_val('Lasso',Lasso(),estimator_scaler,matriz2,rs,blog_X_train,blog_y_train)
matriz2 = estimator_cross_val('Elastic Net',ElasticNet(),estimator_scaler,matriz2,rs,blog_X_train,blog_y_train)
matriz2 = estimator_cross_val('KNN',KNeighborsRegressor(),estimator_scaler,matriz2,rs,blog_X_train,blog_y_train)
matriz2 = estimator_cross_val('CART',DecisionTreeRegressor(),estimator_scaler,matriz2,rs,blog_X_train,blog_y_train)
matriz2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>MAE_mean</th>
      <th>MAE_std</th>
      <th>RMSE_mean</th>
      <th>RMSE_std</th>
      <th>R2_mean</th>
      <th>R2_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression</td>
      <td>9.536043</td>
      <td>0.158151</td>
      <td>30.382693</td>
      <td>1.103323</td>
      <td>0.347973</td>
      <td>0.030809</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ridge Regression</td>
      <td>9.535920</td>
      <td>0.156277</td>
      <td>30.380541</td>
      <td>1.102180</td>
      <td>0.348060</td>
      <td>0.030874</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lasso</td>
      <td>8.435559</td>
      <td>0.157798</td>
      <td>30.499635</td>
      <td>1.170340</td>
      <td>0.343283</td>
      <td>0.025468</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Elastic Net</td>
      <td>8.448594</td>
      <td>0.167776</td>
      <td>30.672138</td>
      <td>1.181716</td>
      <td>0.335904</td>
      <td>0.024106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNN</td>
      <td>6.820830</td>
      <td>0.147086</td>
      <td>29.504949</td>
      <td>1.002954</td>
      <td>0.384447</td>
      <td>0.038376</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CART</td>
      <td>6.379402</td>
      <td>0.323984</td>
      <td>32.991296</td>
      <td>1.777952</td>
      <td>0.228157</td>
      <td>0.086245</td>
    </tr>
  </tbody>
</table>
</div>



-----

### Feature selection

"Feature selection is the process of reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in many cases, to improve the performance of the model" (BROWNLEE, 2021).

There are two main techniques of feature selection: supervised and unsupervised. Supervised methods use the target variable, while unsupervised methods do not (BROWNLEE, 2021).

Besides, the supervised techniques can be divided in (BROWNLEE, 2021):

1. Intrinsic: Algorithms that perform automatic feature selection during training.
2. Wrapper: Search subsets of features that perform according to a predictive model.
3. Filter: Select subsets of features based on their relationship with the target.

### Mutual Information Statistics

Some of the methods of feature selection are more appropriated for numerical variables and others for categorical ones. One popular feature selection techniques used for both numerical variables and categorical variable is Mutual Information Statistics (BROWNLEE, 2021). 

"Mutual information from the field of information theory is the application of information gain (typically used in the construction of decision trees) to feature selection. Mutual information is calculated between two variables and measures the reduction in uncertainty for one variable given a known value of the other variable"  (BROWNLEE, 2021).

We find that many attributes have negligible information value. 181 features have a contribution score over 0.0001, 144 over 0.001, 65 over 0.01, and only 33 over 0.1. **These numbers can vary depending on the training set**. Therefore, we will test the 30, 70, 150, and 190 best features and compare it with the results obtained using all features. 

**We see that the performance using the 70, 150, and 190 best features are almost the same of using all features. Using the 30 beast features is just slighlty worst than using all features**. Moreover, in all cases the KNN model have the best performance.

We could do a grid search to "systematically test a range of different numbers of selected features and discover which results in the best performing model" (BROWNLEE, 2021). **However, a grid search to determine the optimum number of features would require a lot of computing time and the benefit would not be significant in our evaluation**.


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

# feature selection
def select_features(X_train, y_train,k_):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_classif, k=k_) 
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    return X_train_fs, fs
```


```python
# feature selection
blog_X_train_mi, mi = select_features(blog_X_train, blog_y_train,'all')

# what are scores for the features
MI = pd.DataFrame(mi.scores_, columns = ['Score'])

print(MI[MI > 0.0001].count())
print(MI[MI > 0.001].count())
print(MI[MI > 0.01].count())
print(MI[MI > 0.1].count())
```

    Score    192
    dtype: int64
    Score    164
    dtype: int64
    Score    62
    dtype: int64
    Score    33
    dtype: int64



```python
def estimator_cross_val_fea (k,model,estimator,pipe,matriz,rs,X,y):
    pipe_ = pipe(estimator)
    scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error','r2']
    kfold = KFold(n_splits=5, random_state=rs,shuffle=True)
    scores = cross_validate(pipe_,X,y,cv=kfold,scoring=scoring)
    
    mae_scores = -scores.get('test_neg_mean_absolute_error')
    mae_mean = mae_scores.mean()
    mae_std = mae_scores.std()
    
    rmse_scores = -scores.get('test_neg_root_mean_squared_error')
    rmse_mean = rmse_scores.mean()
    rmse_std = rmse_scores.std()
    
    r2_scores = scores.get('test_r2')
    r2_mean = r2_scores.mean()
    r2_std = r2_scores.std()
    
    results_ = [k,model,mae_mean,mae_std,rmse_mean,rmse_std,r2_mean,r2_std]
    results_ = pd.Series(results_, index = matriz.columns)
    results = matriz.append(results_,ignore_index=True)
    return results

matriz_mi = pd.DataFrame(columns=['features','model','MAE_mean','MAE_std','RMSE_mean','RMSE_std','R2_mean','R2_std'])

for k in [30,70,150,190]:

    best_features_mi = MI.transpose()
    best_features_mi.columns = blog_X_train.columns
    best_features_mi.sort_values('Score',axis=1,ascending=False,inplace=True)
    best_features_mi.drop(best_features_mi.iloc[:,k:],axis=1,inplace=True)
    blog_X_train_mi = blog_X_train[best_features_mi.columns]
  
    matriz_mi = estimator_cross_val_fea(k,'Linear Regression',LinearRegression(),     estimator_transf,matriz_mi,rs,blog_X_train_mi,blog_y_train)
    matriz_mi = estimator_cross_val_fea(k,'Ridge Regression', Ridge(),                estimator_transf,matriz_mi,rs,blog_X_train_mi,blog_y_train)
    matriz_mi = estimator_cross_val_fea(k,'Lasso',            Lasso(),                estimator_transf,matriz_mi,rs,blog_X_train_mi,blog_y_train)
    matriz_mi = estimator_cross_val_fea(k,'Elastic Net',      ElasticNet(),           estimator_transf,matriz_mi,rs,blog_X_train_mi,blog_y_train)
    matriz_mi = estimator_cross_val_fea(k,'KNN',              KNeighborsRegressor(),  estimator_transf,matriz_mi,rs,blog_X_train_mi,blog_y_train)
    matriz_mi = estimator_cross_val_fea(k,'CART',             DecisionTreeRegressor(),estimator_transf,matriz_mi,rs,blog_X_train_mi,blog_y_train)

matriz_mi
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>model</th>
      <th>MAE_mean</th>
      <th>MAE_std</th>
      <th>RMSE_mean</th>
      <th>RMSE_std</th>
      <th>R2_mean</th>
      <th>R2_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>Linear Regression</td>
      <td>8.088844</td>
      <td>0.191775</td>
      <td>30.478789</td>
      <td>1.097408</td>
      <td>0.343845</td>
      <td>0.030700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>Ridge Regression</td>
      <td>8.088783</td>
      <td>0.190884</td>
      <td>30.478051</td>
      <td>1.099127</td>
      <td>0.343878</td>
      <td>0.030730</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>Lasso</td>
      <td>8.092827</td>
      <td>0.174670</td>
      <td>30.435213</td>
      <td>1.164739</td>
      <td>0.345830</td>
      <td>0.030713</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>Elastic Net</td>
      <td>8.105448</td>
      <td>0.176620</td>
      <td>30.441817</td>
      <td>1.155065</td>
      <td>0.345539</td>
      <td>0.030518</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30</td>
      <td>KNN</td>
      <td>6.475607</td>
      <td>0.228648</td>
      <td>28.754414</td>
      <td>1.411436</td>
      <td>0.414096</td>
      <td>0.061922</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30</td>
      <td>CART</td>
      <td>7.132292</td>
      <td>0.190374</td>
      <td>34.518267</td>
      <td>1.294679</td>
      <td>0.158520</td>
      <td>0.038165</td>
    </tr>
    <tr>
      <th>6</th>
      <td>70</td>
      <td>Linear Regression</td>
      <td>9.247438</td>
      <td>0.197738</td>
      <td>30.344834</td>
      <td>1.113804</td>
      <td>0.349604</td>
      <td>0.030980</td>
    </tr>
    <tr>
      <th>7</th>
      <td>70</td>
      <td>Ridge Regression</td>
      <td>9.246929</td>
      <td>0.199456</td>
      <td>30.342582</td>
      <td>1.115527</td>
      <td>0.349703</td>
      <td>0.030987</td>
    </tr>
    <tr>
      <th>8</th>
      <td>70</td>
      <td>Lasso</td>
      <td>9.120361</td>
      <td>0.185669</td>
      <td>30.308663</td>
      <td>1.147322</td>
      <td>0.351221</td>
      <td>0.030775</td>
    </tr>
    <tr>
      <th>9</th>
      <td>70</td>
      <td>Elastic Net</td>
      <td>9.133710</td>
      <td>0.187704</td>
      <td>30.310683</td>
      <td>1.145248</td>
      <td>0.351125</td>
      <td>0.030897</td>
    </tr>
    <tr>
      <th>10</th>
      <td>70</td>
      <td>KNN</td>
      <td>6.400156</td>
      <td>0.216946</td>
      <td>28.595075</td>
      <td>1.320200</td>
      <td>0.422814</td>
      <td>0.028841</td>
    </tr>
    <tr>
      <th>11</th>
      <td>70</td>
      <td>CART</td>
      <td>6.515588</td>
      <td>0.416096</td>
      <td>33.741221</td>
      <td>3.170468</td>
      <td>0.193697</td>
      <td>0.119113</td>
    </tr>
    <tr>
      <th>12</th>
      <td>150</td>
      <td>Linear Regression</td>
      <td>9.363710</td>
      <td>0.171935</td>
      <td>30.352465</td>
      <td>1.108021</td>
      <td>0.349276</td>
      <td>0.030831</td>
    </tr>
    <tr>
      <th>13</th>
      <td>150</td>
      <td>Ridge Regression</td>
      <td>9.362735</td>
      <td>0.172976</td>
      <td>30.348999</td>
      <td>1.110113</td>
      <td>0.349427</td>
      <td>0.030846</td>
    </tr>
    <tr>
      <th>14</th>
      <td>150</td>
      <td>Lasso</td>
      <td>9.120361</td>
      <td>0.185669</td>
      <td>30.308663</td>
      <td>1.147322</td>
      <td>0.351221</td>
      <td>0.030775</td>
    </tr>
    <tr>
      <th>15</th>
      <td>150</td>
      <td>Elastic Net</td>
      <td>9.133621</td>
      <td>0.187782</td>
      <td>30.310841</td>
      <td>1.144982</td>
      <td>0.351118</td>
      <td>0.030895</td>
    </tr>
    <tr>
      <th>16</th>
      <td>150</td>
      <td>KNN</td>
      <td>6.398503</td>
      <td>0.215911</td>
      <td>28.586372</td>
      <td>1.309061</td>
      <td>0.423163</td>
      <td>0.028495</td>
    </tr>
    <tr>
      <th>17</th>
      <td>150</td>
      <td>CART</td>
      <td>6.460412</td>
      <td>0.451631</td>
      <td>33.906332</td>
      <td>2.693713</td>
      <td>0.187641</td>
      <td>0.092196</td>
    </tr>
    <tr>
      <th>18</th>
      <td>190</td>
      <td>Linear Regression</td>
      <td>9.417921</td>
      <td>0.175741</td>
      <td>30.362591</td>
      <td>1.110078</td>
      <td>0.348824</td>
      <td>0.031273</td>
    </tr>
    <tr>
      <th>19</th>
      <td>190</td>
      <td>Ridge Regression</td>
      <td>9.416113</td>
      <td>0.176636</td>
      <td>30.358579</td>
      <td>1.112333</td>
      <td>0.348999</td>
      <td>0.031295</td>
    </tr>
    <tr>
      <th>20</th>
      <td>190</td>
      <td>Lasso</td>
      <td>9.120417</td>
      <td>0.185649</td>
      <td>30.308688</td>
      <td>1.147326</td>
      <td>0.351220</td>
      <td>0.030774</td>
    </tr>
    <tr>
      <th>21</th>
      <td>190</td>
      <td>Elastic Net</td>
      <td>9.133701</td>
      <td>0.187659</td>
      <td>30.310884</td>
      <td>1.144988</td>
      <td>0.351117</td>
      <td>0.030894</td>
    </tr>
    <tr>
      <th>22</th>
      <td>190</td>
      <td>KNN</td>
      <td>6.389682</td>
      <td>0.223080</td>
      <td>28.587145</td>
      <td>1.320775</td>
      <td>0.423153</td>
      <td>0.028519</td>
    </tr>
    <tr>
      <th>23</th>
      <td>190</td>
      <td>CART</td>
      <td>6.386770</td>
      <td>0.368084</td>
      <td>33.523654</td>
      <td>2.104213</td>
      <td>0.203756</td>
      <td>0.092511</td>
    </tr>
  </tbody>
</table>
</div>



### Wrapper feature selection method

One way to handle data sets that combines numerical and categorical variables is to use a wrapper method. Some ofent used wrapper methods are Tree-Searching Methods, Stochastic Global Search, Step-Wise Models, and Recursive Feature Elimination (BROWNLEE, 2021).

We use the **Recursive Feature Elimination (RFE) method**. This method searches "for a subset of features by starting with all features in the training dataset and successfully removing features until the desired number remains. This is achieved by fitting the given machine learning algorithm used in the core of the model, ranking features by importance, discarding the least important features, and re-fitting the model" (BROWNLEE, 2021).

We use the RFE to reduce the attributes of the data set. We use the same number of features as in the Mutual information selection method. Thus, we select the 190, 150, 70, and 30 most relevant features and evaluate the models again. 

**We can see that the best model is the KNN with 30 features**. Moreover, the models performed better with the features selected using the RFE than with the ones selected using the mutual information. This model even performed better than the one using all features.


```python
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor

matriz_rfe = pd.DataFrame(columns=['features','model','MAE_mean','MAE_std','RMSE_mean','RMSE_std','R2_mean','R2_std'])
rfe_features = pd.DataFrame()

for k in [30,70,150,190]:
    rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=k)
    rfe.fit(blog_X_train,blog_y_train)
    RF = pd.DataFrame(rfe.support_, columns = ['{} Features'.format(k)])
    rfe_features['{} Features'.format(k)] = RF['{} Features'.format(k)]
    
    best_features_rfe = RF.transpose()
    best_features_rfe.columns = blog_X_train.columns
    best_features_rfe.sort_values('{} Features'.format(k),axis=1,ascending=False,inplace=True)
    best_features_rfe.drop(best_features_rfe.iloc[:,k:],axis=1,inplace=True)
    blog_X_train_rfe = blog_X_train[best_features_rfe.columns]
    blog_X_train_rfe.head()
  
    matriz_rfe = estimator_cross_val_fea(k,'Linear Regression',LinearRegression(),     estimator_transf,matriz_rfe,rs,blog_X_train_rfe,blog_y_train)
    matriz_rfe = estimator_cross_val_fea(k,'Ridge Regression', Ridge(),                estimator_transf,matriz_rfe,rs,blog_X_train_rfe,blog_y_train)
    matriz_rfe = estimator_cross_val_fea(k,'Lasso',            Lasso(),                estimator_transf,matriz_rfe,rs,blog_X_train_rfe,blog_y_train)
    matriz_rfe = estimator_cross_val_fea(k,'Elastic Net',      ElasticNet(),           estimator_transf,matriz_rfe,rs,blog_X_train_rfe,blog_y_train)
    matriz_rfe = estimator_cross_val_fea(k,'KNN',              KNeighborsRegressor(),  estimator_transf,matriz_rfe,rs,blog_X_train_rfe,blog_y_train)
    matriz_rfe = estimator_cross_val_fea(k,'CART',             DecisionTreeRegressor(),estimator_transf,matriz_rfe,rs,blog_X_train_rfe,blog_y_train)

matriz_rfe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>model</th>
      <th>MAE_mean</th>
      <th>MAE_std</th>
      <th>RMSE_mean</th>
      <th>RMSE_std</th>
      <th>R2_mean</th>
      <th>R2_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>Linear Regression</td>
      <td>9.182049</td>
      <td>0.189199</td>
      <td>30.346092</td>
      <td>1.128043</td>
      <td>0.349626</td>
      <td>0.029745</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>Ridge Regression</td>
      <td>9.182005</td>
      <td>0.189197</td>
      <td>30.346085</td>
      <td>1.128039</td>
      <td>0.349627</td>
      <td>0.029745</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>Lasso</td>
      <td>9.087821</td>
      <td>0.193765</td>
      <td>30.345144</td>
      <td>1.146574</td>
      <td>0.349707</td>
      <td>0.029621</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>Elastic Net</td>
      <td>9.116603</td>
      <td>0.192219</td>
      <td>30.347987</td>
      <td>1.150661</td>
      <td>0.349599</td>
      <td>0.029488</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30</td>
      <td>KNN</td>
      <td>6.382809</td>
      <td>0.224521</td>
      <td>28.510587</td>
      <td>1.247672</td>
      <td>0.426304</td>
      <td>0.024114</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30</td>
      <td>CART</td>
      <td>6.495443</td>
      <td>0.336321</td>
      <td>34.154650</td>
      <td>1.970775</td>
      <td>0.174276</td>
      <td>0.081794</td>
    </tr>
    <tr>
      <th>6</th>
      <td>70</td>
      <td>Linear Regression</td>
      <td>9.303662</td>
      <td>0.168855</td>
      <td>30.313320</td>
      <td>1.137163</td>
      <td>0.350973</td>
      <td>0.031427</td>
    </tr>
    <tr>
      <th>7</th>
      <td>70</td>
      <td>Ridge Regression</td>
      <td>9.303296</td>
      <td>0.168870</td>
      <td>30.313111</td>
      <td>1.137050</td>
      <td>0.350982</td>
      <td>0.031425</td>
    </tr>
    <tr>
      <th>8</th>
      <td>70</td>
      <td>Lasso</td>
      <td>9.101846</td>
      <td>0.176443</td>
      <td>30.289602</td>
      <td>1.146506</td>
      <td>0.352024</td>
      <td>0.031049</td>
    </tr>
    <tr>
      <th>9</th>
      <td>70</td>
      <td>Elastic Net</td>
      <td>9.117699</td>
      <td>0.176200</td>
      <td>30.288292</td>
      <td>1.146147</td>
      <td>0.352074</td>
      <td>0.031171</td>
    </tr>
    <tr>
      <th>10</th>
      <td>70</td>
      <td>KNN</td>
      <td>6.357464</td>
      <td>0.214932</td>
      <td>28.582853</td>
      <td>1.256152</td>
      <td>0.423265</td>
      <td>0.026898</td>
    </tr>
    <tr>
      <th>11</th>
      <td>70</td>
      <td>CART</td>
      <td>6.307998</td>
      <td>0.143808</td>
      <td>33.180574</td>
      <td>1.747371</td>
      <td>0.216566</td>
      <td>0.113491</td>
    </tr>
    <tr>
      <th>12</th>
      <td>150</td>
      <td>Linear Regression</td>
      <td>9.430689</td>
      <td>0.173049</td>
      <td>30.367247</td>
      <td>1.117199</td>
      <td>0.348665</td>
      <td>0.030680</td>
    </tr>
    <tr>
      <th>13</th>
      <td>150</td>
      <td>Ridge Regression</td>
      <td>9.427394</td>
      <td>0.174786</td>
      <td>30.366983</td>
      <td>1.117896</td>
      <td>0.348679</td>
      <td>0.030649</td>
    </tr>
    <tr>
      <th>14</th>
      <td>150</td>
      <td>Lasso</td>
      <td>9.139907</td>
      <td>0.183615</td>
      <td>30.307162</td>
      <td>1.136086</td>
      <td>0.351276</td>
      <td>0.030583</td>
    </tr>
    <tr>
      <th>15</th>
      <td>150</td>
      <td>Elastic Net</td>
      <td>9.158194</td>
      <td>0.186132</td>
      <td>30.314601</td>
      <td>1.132900</td>
      <td>0.350944</td>
      <td>0.030752</td>
    </tr>
    <tr>
      <th>16</th>
      <td>150</td>
      <td>KNN</td>
      <td>6.386918</td>
      <td>0.223945</td>
      <td>28.586857</td>
      <td>1.319133</td>
      <td>0.423165</td>
      <td>0.028444</td>
    </tr>
    <tr>
      <th>17</th>
      <td>150</td>
      <td>CART</td>
      <td>6.360462</td>
      <td>0.193760</td>
      <td>32.738433</td>
      <td>2.158942</td>
      <td>0.239203</td>
      <td>0.105038</td>
    </tr>
    <tr>
      <th>18</th>
      <td>190</td>
      <td>Linear Regression</td>
      <td>9.502598</td>
      <td>0.165032</td>
      <td>30.363019</td>
      <td>1.119150</td>
      <td>0.348848</td>
      <td>0.030699</td>
    </tr>
    <tr>
      <th>19</th>
      <td>190</td>
      <td>Ridge Regression</td>
      <td>9.497262</td>
      <td>0.164294</td>
      <td>30.360802</td>
      <td>1.118455</td>
      <td>0.348941</td>
      <td>0.030711</td>
    </tr>
    <tr>
      <th>20</th>
      <td>190</td>
      <td>Lasso</td>
      <td>9.111321</td>
      <td>0.186549</td>
      <td>30.324333</td>
      <td>1.155870</td>
      <td>0.350547</td>
      <td>0.031129</td>
    </tr>
    <tr>
      <th>21</th>
      <td>190</td>
      <td>Elastic Net</td>
      <td>9.132434</td>
      <td>0.187574</td>
      <td>30.324661</td>
      <td>1.145183</td>
      <td>0.350512</td>
      <td>0.031206</td>
    </tr>
    <tr>
      <th>22</th>
      <td>190</td>
      <td>KNN</td>
      <td>6.388018</td>
      <td>0.228146</td>
      <td>28.589889</td>
      <td>1.318046</td>
      <td>0.423055</td>
      <td>0.028165</td>
    </tr>
    <tr>
      <th>23</th>
      <td>190</td>
      <td>CART</td>
      <td>6.407348</td>
      <td>0.277244</td>
      <td>33.224163</td>
      <td>1.969485</td>
      <td>0.216854</td>
      <td>0.094667</td>
    </tr>
  </tbody>
</table>
</div>



### Feature Importance

Another alternative to reduce the number of features is "to score input features using a model and use a filter-based feature selection method. These are called Feature Importance methods" (BROWNLEE, 2021). The most use Feature Importance methods are Classification and Regression Trees (CART), Random Forest, Bagged Decision Trees, and Gradient Boosting.

We use the Random Forest algorithm as our feature importance method. Decision tree algorithms, such as Random Forest, "offer importance scores based on the reduction in the criterion used to select split points, like Gini or entropy" (BROWNLEE, 2021).

We use the same number of features as in the previous selection methods. Thus, we select the 190, 150, 70, and 30 most relevant features and evaluate the models again. We see that the 30 more important features represent almost 80% of all importance. With the 70 most important features we reach 90%, and with the 150 most important 99%. 

**We can see that that KNN was the best model and that it performed similarly for the four cases tested**. Besides, the models performed a little worse with these features than with the features selected using the RFE method.


```python
# random forest for feature importance on a regression problem
from sklearn.ensemble import RandomForestRegressor

# feature selection
def select_features_FI(X_train, y_train,):
    # configure to select all features
    RFR = RandomForestRegressor()
    # learn relationship from training data
    RFR.fit(X_train, y_train)
    # transform train input data
    importance = RFR.feature_importances_
    return importance
```


```python
# feature selection
importance = select_features_FI(blog_X_train, blog_y_train)

# what are scores for the features
FI = pd.DataFrame(importance, columns = ['Importance'])
```


```python
matriz_fi = pd.DataFrame(columns=['features','model','MAE_mean','MAE_std','RMSE_mean','RMSE_std','R2_mean','R2_std'])

for k in [30,70,150,190]:

    best_features_fi = FI.transpose()
    best_features_fi.columns = blog_X_train.columns
    best_features_fi.sort_values('Importance',axis=1,ascending=False,inplace=True)
    best_features_fi.drop(best_features_fi.iloc[:,k:],axis=1,inplace=True)
    blog_X_train_fi = blog_X_train[best_features_fi.columns]
  
    matriz_fi = estimator_cross_val_fea(k,'Linear Regression',LinearRegression(),     estimator_transf,matriz_fi,rs,blog_X_train_fi,blog_y_train)
    matriz_fi = estimator_cross_val_fea(k,'Ridge Regression', Ridge(),                estimator_transf,matriz_fi,rs,blog_X_train_fi,blog_y_train)
    matriz_fi = estimator_cross_val_fea(k,'Lasso',            Lasso(),                estimator_transf,matriz_fi,rs,blog_X_train_fi,blog_y_train)
    matriz_fi = estimator_cross_val_fea(k,'Elastic Net',      ElasticNet(),           estimator_transf,matriz_fi,rs,blog_X_train_fi,blog_y_train)
    matriz_fi = estimator_cross_val_fea(k,'KNN',              KNeighborsRegressor(),  estimator_transf,matriz_fi,rs,blog_X_train_fi,blog_y_train)
    matriz_fi = estimator_cross_val_fea(k,'CART',             DecisionTreeRegressor(),estimator_transf,matriz_fi,rs,blog_X_train_fi,blog_y_train)

matriz_fi
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>model</th>
      <th>MAE_mean</th>
      <th>MAE_std</th>
      <th>RMSE_mean</th>
      <th>RMSE_std</th>
      <th>R2_mean</th>
      <th>R2_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>Linear Regression</td>
      <td>9.208629</td>
      <td>0.197936</td>
      <td>30.305936</td>
      <td>1.129168</td>
      <td>0.351328</td>
      <td>0.030278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>Ridge Regression</td>
      <td>9.185319</td>
      <td>0.192488</td>
      <td>30.311341</td>
      <td>1.122996</td>
      <td>0.351075</td>
      <td>0.030518</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>Lasso</td>
      <td>9.102784</td>
      <td>0.170983</td>
      <td>30.285397</td>
      <td>1.151886</td>
      <td>0.352246</td>
      <td>0.030287</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>Elastic Net</td>
      <td>9.122356</td>
      <td>0.172949</td>
      <td>30.288390</td>
      <td>1.149228</td>
      <td>0.352111</td>
      <td>0.030341</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30</td>
      <td>KNN</td>
      <td>6.385736</td>
      <td>0.223781</td>
      <td>28.531051</td>
      <td>1.311944</td>
      <td>0.425339</td>
      <td>0.029447</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30</td>
      <td>CART</td>
      <td>6.355993</td>
      <td>0.219612</td>
      <td>33.743042</td>
      <td>1.089966</td>
      <td>0.194396</td>
      <td>0.055561</td>
    </tr>
    <tr>
      <th>6</th>
      <td>70</td>
      <td>Linear Regression</td>
      <td>9.295442</td>
      <td>0.184356</td>
      <td>30.330668</td>
      <td>1.131971</td>
      <td>0.350254</td>
      <td>0.030730</td>
    </tr>
    <tr>
      <th>7</th>
      <td>70</td>
      <td>Ridge Regression</td>
      <td>9.288471</td>
      <td>0.186280</td>
      <td>30.331466</td>
      <td>1.134145</td>
      <td>0.350223</td>
      <td>0.030741</td>
    </tr>
    <tr>
      <th>8</th>
      <td>70</td>
      <td>Lasso</td>
      <td>9.113311</td>
      <td>0.176450</td>
      <td>30.291149</td>
      <td>1.156861</td>
      <td>0.351977</td>
      <td>0.031008</td>
    </tr>
    <tr>
      <th>9</th>
      <td>70</td>
      <td>Elastic Net</td>
      <td>9.134904</td>
      <td>0.183125</td>
      <td>30.301134</td>
      <td>1.148820</td>
      <td>0.351541</td>
      <td>0.030901</td>
    </tr>
    <tr>
      <th>10</th>
      <td>70</td>
      <td>KNN</td>
      <td>6.397965</td>
      <td>0.218309</td>
      <td>28.615599</td>
      <td>1.254866</td>
      <td>0.421949</td>
      <td>0.027108</td>
    </tr>
    <tr>
      <th>11</th>
      <td>70</td>
      <td>CART</td>
      <td>6.461393</td>
      <td>0.293485</td>
      <td>33.675495</td>
      <td>1.977965</td>
      <td>0.199257</td>
      <td>0.061369</td>
    </tr>
    <tr>
      <th>12</th>
      <td>150</td>
      <td>Linear Regression</td>
      <td>9.444099</td>
      <td>0.167406</td>
      <td>30.374028</td>
      <td>1.117694</td>
      <td>0.348379</td>
      <td>0.030588</td>
    </tr>
    <tr>
      <th>13</th>
      <td>150</td>
      <td>Ridge Regression</td>
      <td>9.440118</td>
      <td>0.168038</td>
      <td>30.371687</td>
      <td>1.117332</td>
      <td>0.348478</td>
      <td>0.030623</td>
    </tr>
    <tr>
      <th>14</th>
      <td>150</td>
      <td>Lasso</td>
      <td>9.110817</td>
      <td>0.185537</td>
      <td>30.304423</td>
      <td>1.138442</td>
      <td>0.351389</td>
      <td>0.030751</td>
    </tr>
    <tr>
      <th>15</th>
      <td>150</td>
      <td>Elastic Net</td>
      <td>9.133604</td>
      <td>0.185830</td>
      <td>30.310303</td>
      <td>1.137118</td>
      <td>0.351133</td>
      <td>0.030805</td>
    </tr>
    <tr>
      <th>16</th>
      <td>150</td>
      <td>KNN</td>
      <td>6.387911</td>
      <td>0.228185</td>
      <td>28.589800</td>
      <td>1.317743</td>
      <td>0.423058</td>
      <td>0.028159</td>
    </tr>
    <tr>
      <th>17</th>
      <td>150</td>
      <td>CART</td>
      <td>6.426588</td>
      <td>0.217897</td>
      <td>33.181804</td>
      <td>1.640639</td>
      <td>0.216888</td>
      <td>0.103949</td>
    </tr>
    <tr>
      <th>18</th>
      <td>190</td>
      <td>Linear Regression</td>
      <td>9.493089</td>
      <td>0.167189</td>
      <td>30.390227</td>
      <td>1.115748</td>
      <td>0.347675</td>
      <td>0.030684</td>
    </tr>
    <tr>
      <th>19</th>
      <td>190</td>
      <td>Ridge Regression</td>
      <td>9.490867</td>
      <td>0.169898</td>
      <td>30.387115</td>
      <td>1.117990</td>
      <td>0.347810</td>
      <td>0.030740</td>
    </tr>
    <tr>
      <th>20</th>
      <td>190</td>
      <td>Lasso</td>
      <td>9.110817</td>
      <td>0.185537</td>
      <td>30.304423</td>
      <td>1.138442</td>
      <td>0.351389</td>
      <td>0.030751</td>
    </tr>
    <tr>
      <th>21</th>
      <td>190</td>
      <td>Elastic Net</td>
      <td>9.133602</td>
      <td>0.185821</td>
      <td>30.310496</td>
      <td>1.136838</td>
      <td>0.351124</td>
      <td>0.030802</td>
    </tr>
    <tr>
      <th>22</th>
      <td>190</td>
      <td>KNN</td>
      <td>6.388277</td>
      <td>0.227648</td>
      <td>28.589934</td>
      <td>1.317930</td>
      <td>0.423053</td>
      <td>0.028167</td>
    </tr>
    <tr>
      <th>23</th>
      <td>190</td>
      <td>CART</td>
      <td>6.406261</td>
      <td>0.282191</td>
      <td>33.274056</td>
      <td>1.947994</td>
      <td>0.211473</td>
      <td>0.117220</td>
    </tr>
  </tbody>
</table>
</div>



### Comparing the features

Finally, we compare the features selected by each model. We see that the percentage of features selected by all methods increase as the number of features used increases:

1. 30 best - 9 shared (30.0%)
2. 70 best - 32 shared (45.7%)
3. 150 best - 92 shared (61.3%)
4. 190 best - 134 shared (70.5%)

Besides, we can argue that the 9 features that were selected by all methods as one of the 30 most relevant fatures are the most significant ones for predicting our target variable.


```python
MI_30 = MI.sort_values(by=['Score'],ascending=False)[:30].reset_index()
FI_30 = FI.sort_values(by=['Importance'],ascending=False)[:30].reset_index()
RF_30 = rfe_features.sort_values(by=['30 Features'],ascending=False)[:30].reset_index()
RF_30 = RF_30.drop(columns=['70 Features','150 Features','190 Features'])
merged_30 = pd.merge(FI_30, MI_30, on=['index'], how='inner')
merged_30 = pd.merge(merged_30, RF_30, on=['index'], how='inner')
len(merged_30)
```




    9




```python
MI_70 = MI.sort_values(by=['Score'],ascending=False)[:70].reset_index()
FI_70 = FI.sort_values(by=['Importance'],ascending=False)[:70].reset_index()
RF_70 = rfe_features.sort_values(by=['70 Features'],ascending=False)[:70].reset_index()
RF_70 = RF_70.drop(columns=['30 Features','150 Features','190 Features'])
merged_70 = pd.merge(FI_70, MI_70, on=['index'], how='inner')
merged_70 = pd.merge(merged_70, RF_70, on=['index'], how='inner')
len(merged_70)
```




    32




```python
MI_150 = MI.sort_values(by=['Score'],ascending=False)[:150].reset_index()
FI_150 = FI.sort_values(by=['Importance'],ascending=False)[:150].reset_index()
RF_150 = rfe_features.sort_values(by=['150 Features'],ascending=False)[:150].reset_index()
RF_150 = RF_150.drop(columns=['30 Features','70 Features','190 Features'])
merged_150 = pd.merge(FI_150, MI_150, on=['index'], how='inner')
merged_150 = pd.merge(merged_150, RF_150, on=['index'], how='inner')
len(merged_150)
```




    92




```python
MI_190 = MI.sort_values(by=['Score'],ascending=False)[:190].reset_index()
FI_190 = FI.sort_values(by=['Importance'],ascending=False)[:190].reset_index()
RF_190 = rfe_features.sort_values(by=['190 Features'],ascending=False)[:190].reset_index()
RF_190 = RF_190.drop(columns=['30 Features','70 Features','150 Features'])
merged_190 = pd.merge(FI_190, MI_190, on=['index'], how='inner')
merged_190 = pd.merge(merged_190, RF_190, on=['index'], how='inner')
len(merged_190)
```




    134



----------

### Dimensionality reduction

"Dimensionality reduction refers to techniques that reduce the number of input variables in a dataset. Fewer input dimensions often mean correspondingly fewer parameters or a simpler structure in the machine learning model, referred to as degrees of freedom. A model with too many degrees of freedom is likely to overfit the training dataset and therefore may not perform well on new data" (BROWNLEE, 2021).

There are several techniques to reduce a data set dimensionality. **In this notebook, we use the Principal Component Analysis (PCA), which is the most used method for dimensionality reduction**. "It can be thought of as a projection method where data with m-columns (features) is projected into a subspace with m or fewer columns, whilst retaining the essence of the original data" (BROWNLEE, 2021). 

We reduce the dimensionality of the data set using the same number of features used in the feature selection section, 30, 70, 150, and 190. **We verify that the results are quite similar to the ones we obtained in the feature selection and that the level of the dimensionality reduction have little influence in the results**. Besides, once more the KNN model is better than the other models tested.


```python
from sklearn.decomposition import PCA

def estimator_pca(estimator,k):
    #imputer = SimpleImputer(strategy='median')
    pipeline = Pipeline(steps=[('pca',PCA(n_components=k)),('model', estimator)])
    return pipeline 

def estimator_cross_val_pca(k,model,estimator,pipe,matriz,rs,X,y):
    pipe_ = pipe(estimator,k)
    scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error','r2']
    kfold = KFold(n_splits=5, random_state=rs,shuffle=True)
    scores = cross_validate(pipe_,X,y,cv=kfold,scoring=scoring)
    
    mae_scores = -scores.get('test_neg_mean_absolute_error')
    mae_mean = mae_scores.mean()
    mae_std = mae_scores.std()
    
    rmse_scores = -scores.get('test_neg_root_mean_squared_error')
    rmse_mean = rmse_scores.mean()
    rmse_std = rmse_scores.std()
    
    r2_scores = scores.get('test_r2')
    r2_mean = r2_scores.mean()
    r2_std = r2_scores.std()
    
    results_ = [k,model,mae_mean,mae_std,rmse_mean,rmse_std,r2_mean,r2_std]
    results_ = pd.Series(results_, index = matriz.columns)
    results = matriz.append(results_,ignore_index=True)
    return results

matriz_pca = pd.DataFrame(columns=['dimensionality','model','MAE_mean','MAE_std','RMSE_mean','RMSE_std','R2_mean','R2_std'])

for k in [30,70,150,190]:
  
    matriz_pca = estimator_cross_val_pca(k,'Linear Regression',LinearRegression(),     estimator_pca,matriz_pca,rs,blog_X_train,blog_y_train)
    matriz_pca = estimator_cross_val_pca(k,'Ridge Regression', Ridge(),                estimator_pca,matriz_pca,rs,blog_X_train,blog_y_train)
    matriz_pca = estimator_cross_val_pca(k,'Lasso',            Lasso(),                estimator_pca,matriz_pca,rs,blog_X_train,blog_y_train)
    matriz_pca = estimator_cross_val_pca(k,'Elastic Net',      ElasticNet(),           estimator_pca,matriz_pca,rs,blog_X_train,blog_y_train)
    matriz_pca = estimator_cross_val_pca(k,'KNN',              KNeighborsRegressor(),  estimator_pca,matriz_pca,rs,blog_X_train,blog_y_train)
    matriz_pca = estimator_cross_val_pca(k,'CART',             DecisionTreeRegressor(),estimator_pca,matriz_pca,rs,blog_X_train,blog_y_train)

matriz_pca
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dimensionality</th>
      <th>model</th>
      <th>MAE_mean</th>
      <th>MAE_std</th>
      <th>RMSE_mean</th>
      <th>RMSE_std</th>
      <th>R2_mean</th>
      <th>R2_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>Linear Regression</td>
      <td>9.170274</td>
      <td>0.196680</td>
      <td>30.337866</td>
      <td>1.132621</td>
      <td>0.349924</td>
      <td>0.031216</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>Ridge Regression</td>
      <td>9.170276</td>
      <td>0.196676</td>
      <td>30.337863</td>
      <td>1.132618</td>
      <td>0.349925</td>
      <td>0.031216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>Lasso</td>
      <td>9.122299</td>
      <td>0.180373</td>
      <td>30.305504</td>
      <td>1.149764</td>
      <td>0.351367</td>
      <td>0.030623</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>Elastic Net</td>
      <td>9.145455</td>
      <td>0.185224</td>
      <td>30.312383</td>
      <td>1.147534</td>
      <td>0.351061</td>
      <td>0.030783</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30</td>
      <td>KNN</td>
      <td>6.391033</td>
      <td>0.215455</td>
      <td>28.586793</td>
      <td>1.320662</td>
      <td>0.423153</td>
      <td>0.028792</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30</td>
      <td>CART</td>
      <td>6.667366</td>
      <td>0.249153</td>
      <td>34.565825</td>
      <td>1.622042</td>
      <td>0.151668</td>
      <td>0.099029</td>
    </tr>
    <tr>
      <th>6</th>
      <td>70</td>
      <td>Linear Regression</td>
      <td>9.291957</td>
      <td>0.198461</td>
      <td>30.338712</td>
      <td>1.113666</td>
      <td>0.349879</td>
      <td>0.030748</td>
    </tr>
    <tr>
      <th>7</th>
      <td>70</td>
      <td>Ridge Regression</td>
      <td>9.292803</td>
      <td>0.200156</td>
      <td>30.337724</td>
      <td>1.111993</td>
      <td>0.349920</td>
      <td>0.030711</td>
    </tr>
    <tr>
      <th>8</th>
      <td>70</td>
      <td>Lasso</td>
      <td>9.122299</td>
      <td>0.180373</td>
      <td>30.305504</td>
      <td>1.149764</td>
      <td>0.351367</td>
      <td>0.030623</td>
    </tr>
    <tr>
      <th>9</th>
      <td>70</td>
      <td>Elastic Net</td>
      <td>9.144900</td>
      <td>0.185320</td>
      <td>30.311839</td>
      <td>1.147532</td>
      <td>0.351086</td>
      <td>0.030759</td>
    </tr>
    <tr>
      <th>10</th>
      <td>70</td>
      <td>KNN</td>
      <td>6.386980</td>
      <td>0.224841</td>
      <td>28.576718</td>
      <td>1.326137</td>
      <td>0.423587</td>
      <td>0.028486</td>
    </tr>
    <tr>
      <th>11</th>
      <td>70</td>
      <td>CART</td>
      <td>6.879595</td>
      <td>0.118472</td>
      <td>34.956786</td>
      <td>0.753124</td>
      <td>0.132987</td>
      <td>0.084415</td>
    </tr>
    <tr>
      <th>12</th>
      <td>150</td>
      <td>Linear Regression</td>
      <td>9.459503</td>
      <td>0.168487</td>
      <td>30.358749</td>
      <td>1.116275</td>
      <td>0.349027</td>
      <td>0.030679</td>
    </tr>
    <tr>
      <th>13</th>
      <td>150</td>
      <td>Ridge Regression</td>
      <td>9.458508</td>
      <td>0.169914</td>
      <td>30.359304</td>
      <td>1.116341</td>
      <td>0.349005</td>
      <td>0.030642</td>
    </tr>
    <tr>
      <th>14</th>
      <td>150</td>
      <td>Lasso</td>
      <td>9.122299</td>
      <td>0.180373</td>
      <td>30.305504</td>
      <td>1.149764</td>
      <td>0.351367</td>
      <td>0.030623</td>
    </tr>
    <tr>
      <th>15</th>
      <td>150</td>
      <td>Elastic Net</td>
      <td>9.144900</td>
      <td>0.185320</td>
      <td>30.311839</td>
      <td>1.147532</td>
      <td>0.351086</td>
      <td>0.030759</td>
    </tr>
    <tr>
      <th>16</th>
      <td>150</td>
      <td>KNN</td>
      <td>6.392342</td>
      <td>0.226349</td>
      <td>28.583801</td>
      <td>1.326865</td>
      <td>0.423308</td>
      <td>0.028381</td>
    </tr>
    <tr>
      <th>17</th>
      <td>150</td>
      <td>CART</td>
      <td>6.962126</td>
      <td>0.287425</td>
      <td>35.071667</td>
      <td>1.803805</td>
      <td>0.132001</td>
      <td>0.048294</td>
    </tr>
    <tr>
      <th>18</th>
      <td>190</td>
      <td>Linear Regression</td>
      <td>9.504434</td>
      <td>0.161015</td>
      <td>30.368124</td>
      <td>1.113705</td>
      <td>0.348606</td>
      <td>0.031003</td>
    </tr>
    <tr>
      <th>19</th>
      <td>190</td>
      <td>Ridge Regression</td>
      <td>9.504412</td>
      <td>0.160918</td>
      <td>30.368032</td>
      <td>1.113380</td>
      <td>0.348609</td>
      <td>0.031009</td>
    </tr>
    <tr>
      <th>20</th>
      <td>190</td>
      <td>Lasso</td>
      <td>9.122299</td>
      <td>0.180373</td>
      <td>30.305504</td>
      <td>1.149764</td>
      <td>0.351367</td>
      <td>0.030623</td>
    </tr>
    <tr>
      <th>21</th>
      <td>190</td>
      <td>Elastic Net</td>
      <td>9.144900</td>
      <td>0.185320</td>
      <td>30.311839</td>
      <td>1.147532</td>
      <td>0.351086</td>
      <td>0.030759</td>
    </tr>
    <tr>
      <th>22</th>
      <td>190</td>
      <td>KNN</td>
      <td>6.390472</td>
      <td>0.224537</td>
      <td>28.581474</td>
      <td>1.325347</td>
      <td>0.423397</td>
      <td>0.028422</td>
    </tr>
    <tr>
      <th>23</th>
      <td>190</td>
      <td>CART</td>
      <td>7.055118</td>
      <td>0.209190</td>
      <td>35.846981</td>
      <td>1.809665</td>
      <td>0.091019</td>
      <td>0.082416</td>
    </tr>
  </tbody>
</table>
</div>



---------

# 3. Evaluate the ML model

Now evaluate the performance of our ML model in the test set, to see how it perform with unseen data.

We will do two tests. In the first one we use the KNN model and the 30 features selected using the RFE method. For the second test, we use the KNN model and the data set reduced using the PCA method.

First, we import the test set.

After testing the models, we verify that the performance of our model with the test set is similar to the performance with the train set. The MAE and RMSE are actually a little better but the R² is lower. Besides, the RMSE is considrably higher than the MAE. This result suggests that our data has many outliers and, consequently, our model is making some big errors.

Finally, we see that using the features selected by the RFE method and doing a dimensionality reduction using the PCA have similiar results.

### Getting the test set


```python
import os
import glob

os.chdir(r"/Users/leuzinger/Dropbox/Data Science/Awari/Regressions/BlogFeedback/Test/")
filenames = [i for i in glob.glob("*.csv")]
df = [pd.read_csv(file, sep = ",", header=None,) 
      for file in filenames]
```


```python
blog_test = df[0]

for i in range(1,len(df)):
    blog_test = blog_test.append(df[i]) 

blog_test.reset_index(drop=True,inplace=True)
blog_test.set_axis(att,axis=1,inplace=True)
blog_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>blog_avg_total</th>
      <th>blog_std_total</th>
      <th>blog_min_total</th>
      <th>blog_max_total</th>
      <th>blog_median_total</th>
      <th>blog_avg_last24h</th>
      <th>blog_std_last24h</th>
      <th>blog_min_last24h</th>
      <th>blog_max_last24h</th>
      <th>blog_median_last24h</th>
      <th>blog_avg_24-48h</th>
      <th>blog_std_24-48h</th>
      <th>blog_min_24-48h</th>
      <th>blog_max_24-48h</th>
      <th>blog_median_24-48h</th>
      <th>blog_avg_first24h</th>
      <th>blog_std_first24h</th>
      <th>blog_min_first24h</th>
      <th>blog_max_first24h</th>
      <th>blog_median_first24h</th>
      <th>blog_avg_difference</th>
      <th>blog_std_difference</th>
      <th>blog_min_difference</th>
      <th>blog_max_difference</th>
      <th>blog_median_difference</th>
      <th>blog_avg_total_tr</th>
      <th>blog_std_total_tr</th>
      <th>blog_min_total_tr</th>
      <th>blog_max_total_tr</th>
      <th>blog_median_total_tr</th>
      <th>blog_avg_last24h_tr</th>
      <th>blog_std_last24h_tr</th>
      <th>blog_min_last24h_tr</th>
      <th>blog_max_last24h_tr</th>
      <th>blog_median_last24h_tr</th>
      <th>blog_avg_24-48h_tr</th>
      <th>blog_std_24-48h_tr</th>
      <th>blog_min_24-48h_tr</th>
      <th>blog_max_24-48h_tr</th>
      <th>blog_median_24-48h_tr</th>
      <th>blog_avg_first24h_tr</th>
      <th>blog_std_first24h_tr</th>
      <th>blog_min_first24h_tr</th>
      <th>blog_max_first24h_tr</th>
      <th>blog_median_first24h_tr</th>
      <th>blog_avg_difference_tr</th>
      <th>blog_std_difference_tr</th>
      <th>blog_min_difference_tr</th>
      <th>blog_max_difference_tr</th>
      <th>blog_median_difference_tr</th>
      <th>total</th>
      <th>last24h</th>
      <th>24-48h</th>
      <th>first24h</th>
      <th>difference</th>
      <th>total_tr</th>
      <th>last24h_tr</th>
      <th>24-48h_tr</th>
      <th>first24h_tr</th>
      <th>difference_tr</th>
      <th>time_first_post</th>
      <th>lenght_post</th>
      <th>word1</th>
      <th>word2</th>
      <th>word3</th>
      <th>word4</th>
      <th>word5</th>
      <th>word6</th>
      <th>word7</th>
      <th>word8</th>
      <th>word9</th>
      <th>word10</th>
      <th>word11</th>
      <th>word12</th>
      <th>word13</th>
      <th>word14</th>
      <th>word15</th>
      <th>word16</th>
      <th>word17</th>
      <th>word18</th>
      <th>word19</th>
      <th>word20</th>
      <th>word21</th>
      <th>word22</th>
      <th>word23</th>
      <th>word24</th>
      <th>word25</th>
      <th>word26</th>
      <th>word27</th>
      <th>word28</th>
      <th>word29</th>
      <th>word30</th>
      <th>word31</th>
      <th>word32</th>
      <th>word33</th>
      <th>word34</th>
      <th>word35</th>
      <th>word36</th>
      <th>word37</th>
      <th>word38</th>
      <th>word39</th>
      <th>word40</th>
      <th>word41</th>
      <th>word42</th>
      <th>word43</th>
      <th>word44</th>
      <th>word45</th>
      <th>word46</th>
      <th>word47</th>
      <th>word48</th>
      <th>word49</th>
      <th>word50</th>
      <th>word51</th>
      <th>word52</th>
      <th>word53</th>
      <th>word54</th>
      <th>word55</th>
      <th>word56</th>
      <th>word57</th>
      <th>word58</th>
      <th>word59</th>
      <th>word60</th>
      <th>word61</th>
      <th>word62</th>
      <th>word63</th>
      <th>word64</th>
      <th>word65</th>
      <th>word66</th>
      <th>word67</th>
      <th>word68</th>
      <th>word69</th>
      <th>word70</th>
      <th>word71</th>
      <th>word72</th>
      <th>word73</th>
      <th>word74</th>
      <th>word75</th>
      <th>word76</th>
      <th>word77</th>
      <th>word78</th>
      <th>word79</th>
      <th>word80</th>
      <th>word81</th>
      <th>word82</th>
      <th>word83</th>
      <th>word84</th>
      <th>word85</th>
      <th>word86</th>
      <th>word87</th>
      <th>word88</th>
      <th>word89</th>
      <th>word90</th>
      <th>word91</th>
      <th>word92</th>
      <th>word93</th>
      <th>word94</th>
      <th>word95</th>
      <th>word96</th>
      <th>word97</th>
      <th>word98</th>
      <th>word99</th>
      <th>word100</th>
      <th>word101</th>
      <th>word102</th>
      <th>word103</th>
      <th>word104</th>
      <th>word105</th>
      <th>word106</th>
      <th>word107</th>
      <th>word108</th>
      <th>word109</th>
      <th>word110</th>
      <th>word111</th>
      <th>word112</th>
      <th>word113</th>
      <th>word114</th>
      <th>word115</th>
      <th>word116</th>
      <th>word117</th>
      <th>word118</th>
      <th>word119</th>
      <th>word120</th>
      <th>word121</th>
      <th>word122</th>
      <th>word123</th>
      <th>word124</th>
      <th>word125</th>
      <th>word126</th>
      <th>word127</th>
      <th>word128</th>
      <th>word129</th>
      <th>word130</th>
      <th>word131</th>
      <th>word132</th>
      <th>word133</th>
      <th>word134</th>
      <th>word135</th>
      <th>word136</th>
      <th>word137</th>
      <th>word138</th>
      <th>word139</th>
      <th>word140</th>
      <th>word141</th>
      <th>word142</th>
      <th>word143</th>
      <th>word144</th>
      <th>word145</th>
      <th>word146</th>
      <th>word147</th>
      <th>word148</th>
      <th>word149</th>
      <th>word150</th>
      <th>word151</th>
      <th>word152</th>
      <th>word153</th>
      <th>word154</th>
      <th>word155</th>
      <th>word156</th>
      <th>word157</th>
      <th>word158</th>
      <th>word159</th>
      <th>word160</th>
      <th>word161</th>
      <th>word162</th>
      <th>word163</th>
      <th>word164</th>
      <th>word165</th>
      <th>word166</th>
      <th>word167</th>
      <th>word168</th>
      <th>word169</th>
      <th>word170</th>
      <th>word171</th>
      <th>word172</th>
      <th>word173</th>
      <th>word174</th>
      <th>word175</th>
      <th>word176</th>
      <th>word177</th>
      <th>word178</th>
      <th>word179</th>
      <th>word180</th>
      <th>word181</th>
      <th>word182</th>
      <th>word183</th>
      <th>word184</th>
      <th>word185</th>
      <th>word186</th>
      <th>word187</th>
      <th>word188</th>
      <th>word189</th>
      <th>word190</th>
      <th>word191</th>
      <th>word192</th>
      <th>word193</th>
      <th>word194</th>
      <th>word195</th>
      <th>word196</th>
      <th>word197</th>
      <th>word198</th>
      <th>word199</th>
      <th>word200</th>
      <th>Mon_bl</th>
      <th>Tue_bl</th>
      <th>Wed_bl</th>
      <th>Thu_bl</th>
      <th>Fri_bl</th>
      <th>Sat_bl</th>
      <th>Sun_bl</th>
      <th>Mon_post</th>
      <th>Tue_post</th>
      <th>Wed_post</th>
      <th>Thu_post</th>
      <th>Fri_post</th>
      <th>Sat_post</th>
      <th>Sun_post</th>
      <th>parent_pages</th>
      <th>min_parent</th>
      <th>max_parent</th>
      <th>avg_parent</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.064516</td>
      <td>0.24567</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.176685</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.176685</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.254</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>1470.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>102.0</td>
      <td>91.0</td>
      <td>11.0</td>
      <td>101.0</td>
      <td>80.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>27.0</td>
      <td>3520.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.056075</td>
      <td>0.330159</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.018692</td>
      <td>0.192442</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.018692</td>
      <td>0.192442</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.056075</td>
      <td>0.330159</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.273434</td>
      <td>-2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>800.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.064516</td>
      <td>0.24567</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.176685</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.176685</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.254</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>51.0</td>
      <td>1468.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47.776787</td>
      <td>93.737470</td>
      <td>1.0</td>
      <td>598.0</td>
      <td>7.5</td>
      <td>17.857143</td>
      <td>56.888218</td>
      <td>0.0</td>
      <td>594.0</td>
      <td>1.0</td>
      <td>17.350447</td>
      <td>56.911470</td>
      <td>0.0</td>
      <td>594.0</td>
      <td>1.0</td>
      <td>46.386160</td>
      <td>91.284140</td>
      <td>1.0</td>
      <td>595.0</td>
      <td>7.0</td>
      <td>0.506696</td>
      <td>79.062050</td>
      <td>-590.0</td>
      <td>594.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
blog_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7624 entries, 0 to 7623
    Columns: 281 entries, blog_avg_total to target
    dtypes: float64(281)
    memory usage: 16.3 MB



```python
blog_X_test = blog_test.drop('target',axis=1).copy()
blog_y_test = blog_test['target'].copy()
```

### Evaluating the ML models


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=30) 
knn = KNeighborsRegressor()
pipe_rfe = Pipeline(steps=[('rfe',rfe),('knn',knn)])

pipe_rfe.fit(blog_X_train,blog_y_train)
blog_y_hat = pipe_rfe.predict(blog_X_test)

final_mae = mean_absolute_error(blog_y_test,blog_y_hat)
final_mse = mean_squared_error(blog_y_test,blog_y_hat)
final_rmse = np.sqrt(final_mse)
final_r2 = r2_score(blog_y_test,blog_y_hat)
print('MAE:  %.2f'%final_mae,'\nRMSE: %.2f'%final_rmse,'\nR2:   %.2f'%final_r2)
```

    MAE:  5.79 
    RMSE: 25.12 
    R2:   0.32



```python
pipe_pca = Pipeline(steps=[('pca',PCA(n_components=30)),('knn', KNeighborsRegressor())])
pipe_pca.fit(blog_X_train,blog_y_train)
blog_y_hat = pipe_pca.predict(blog_X_test)

final_mae = mean_absolute_error(blog_y_test,blog_y_hat)
final_mse = mean_squared_error(blog_y_test,blog_y_hat)
final_rmse = np.sqrt(final_mse)
final_r2 = r2_score(blog_y_test,blog_y_hat)
print('MAE:  %.2f'%final_mae,'\nRMSE: %.2f'%final_rmse,'\nR2:   %.2f'%final_r2)
```

    MAE:  5.72 
    RMSE: 25.10 
    R2:   0.32


----------------------

## 4. Conclusion

In this notebook, we created a model to predict the number of blog posts in the next 24h based on several attributes of the post. First, we tested some regression models: 

1. Linear regression
2. Ridge regression
3. Lasso regression
4. Elastic Net
5. Classification and Regression Trees (CART)
6. k-Nearest Neighbors (KNN)

In this first tests, the KNN was the best performing method.

However, we verified that the large number of features in our data was demanding a high computing time to run the models. Therefore, we tested some techniques to reduce the number of features:

1. Mutual Information Statistics
2. Recursive Feature Elimination (RFE)
3. Random Forest

The features selected by the RFE were the ones that resulted in the best performance of the KNN model.

Finaly, we also used a dimensionality reduction method, the Principal Component Analysis (PCA) to reduce the size of our data set. Our results with the train set showed that both the RFE and the PCA, combined with the KNN model, had similar results.

**Therefore, we tested two models with our test set: (i) KNN + RFE and (ii) KNN + PCA. We verified that the models performed almost identically**.

**However, our models performed modestly at best. All evaluation metrics used are poor**, especially the RMSE and the R². The fact that the RMSE is high suggests that our data has many outliers and, consequently, our model is making some big errors. **Nonetheless, given that these are quite simple regression methods, we could consider that the results are reasonable**. More complex models could be used to achieve better predictions. However, these models would probably demand more time to build and more computing power to run, which could actually mean a worse cost-benefit.
