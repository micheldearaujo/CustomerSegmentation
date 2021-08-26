#   Machine Learning and Data Science Project
#   
#   Customer Segmentation
#
#   @author   micheldearaujo
#   created at    2021 ago 24   13:03 -3:00 GMT

# dataset: https://drive.google.com/file/d/19BOhwz52NUY3dg8XErVYglctpr5sjTy4/view

# In this Data Science/ Machine Learning project we are going to perform some customer segmentation.
# Customer Segmentation is one of the most common practices one it comes to client analysis and clustering
# Projects.

# Regarding the algorithms that will be used in this project, we are going to test two different algorithms:
# the K-Means and the KNN (K-Nearest Neighbors)
library(RColorBrewer)
library(dplyr)
# ========================== Data Exploration =========================== #

# --------- Reading the data --------------
# Let's start from the beginning and read the data.
# The dataset comes in a csv format, so we just need to use the read.csv() function

customer <- read.csv('data/Mall_Customers.csv')
View(customer)
str(customer)

# This dataset contains 200 observations of 5 different features: CustomerID, Gender, age
# Annual Income and Spending Score. It is a really small dataset but can provides us with a nice
# exercise on building models.

# Let's take a look in a simple summary of the data
summary(customer)

# This seems to be a tidy dataset. We have a wide range of ages, incomes and spending score.
# This is will ensure that our model will not have some kind of bias lead by data homogeneity.

# What looks the distribution of the gender occurrences? To answer that question
# We can use a simple bar plot.

genders <- table(customer$Gender)
barplot(genders,
        main='Gender Distribution of Customers database',
        xlab='Gender',
        ylab='# of Customers',
        col=brewer.pal(name='Dark2', n=2),
        legend=rownames(genders))


# This barplot shows us that women are buy a little bit more than men.

# What about the age distribution? We can see the overal age distribution
# and the age distribution by gender as well.

hist(customer$Age,
     col='steelblue',
     xlab='Age',
     ylab='# of Customers',
     labels=TRUE)

# Are there outliers? We can easily check that information by using a box plot

boxplot(customer$Age ~ customer$Gender,
        ylab='Age',
        xlab='Sex',
        main='Age distribution by sex',
        col=brewer.pal(2, 'Dark2'))

# The age amplitude for men are a little big larger than women's.

# ---- Annual Income
# Now let's take a look about the annual income distribution by gender and by age
# Plotting the income distribution by gender
par(mfrow=c(1,2))

hist(filter(customer, Gender=='Female')$Annual.Income..k..,
     xlab='Annual Income (x1000)',
     ylab='# of customers',
     main='Income Distribution for Female',
     col='steelblue')

hist(filter(customer, Gender=='Male')$Annual.Income..k..,
     xlab='Annual Incone (x1000)',
     ylab='# of Customers',
     main='Income Distribution for Male',
     col='steelblue')

# And plotting the distribution of income as function of age
customer$Gender <- as.factor(customer$Gender)
par(mfrow=c(1,1))
plot(x=customer$Age,
        y=customer$Spending.Score..1.100.,
     pch=16,
     cex=1.5,
     col=customer$Gender,
     xlab='Age',
     ylab='Annual Income (x1000)',
     main='Income Distribution by age and sex')

legend('topright', legend=unique(customer$Gender),
       col=c('black', 'red'),
       pch=16)


# ------- Spending Score
# Now let us take a look at the distribution of the speding score
# of our customers. For a start, we can make a histogram together with a boxplot

par(mfrow=c(1,2))

hist(customer$Spending.Score..1.100.,
     main='Speding Score Histogram',
     xlab='Speding Score',
     ylab='# Of Customers',
     col='steelblue')
boxplot(customer$Spending.Score..1.100. ~ customer$Gender,
        main='Speding Score Distribution',
        xlab='Gender',
        ylab='Spending Score',
        col=brewer.pal(2, 'Dark2'))

# Interesting result. The histogram shows that the majority of customers
# have a median spending score, followed by customers who has or higher or lower
# and the minor number of customers are in the first and third quantile.

# ================== Clustering the Customers =================#

# --------- K-means Algorithm

# The first step to use the K-means clustering algorithm is to setup the number of cluster K that we wish 
# to produce the final output.

# Summing up the K-means clustering –
# 
# 1. We specify the number of clusters that we need to create.
# 2. The algorithm selects k objects at random from the dataset. This object is the initial cluster or mean.
# The closest centroid obtains the assignment of a new observation. We base this assignment on the
# Euclidean Distance between object and the centroid.
# 3. k clusters in the data points update the centroid through calculation of the new mean values present
# in all the data points of the cluster. The kth cluster’s centroid has a length of p that contains means
# of all variables for observations in the k-th cluster. We denote the number of variables with p.
# 4. Iterative minimization of the total within the sum of squares. Then through the iterative minimization of
# the total sum of the square, the assignment stop wavering when we achieve maximum iteration.
# The default value is 10 that the R software uses for the maximum iterations.

library(purrr)
set.seed(123)

## ------ Elbow Method
# Create a function to calculate total intra-cluster sum of squares
iss <- function(k) {
  kmeans(x=customer[, 3:5],
         centers = k,
         iter.max=100,
         nstart=100,
         algorithm='Lloyd')$tot.withinss
}

# Determine the number of clusters that we want to test
k.values <- 1:10

# Use the map_dbl function to map all the k.values to the function ISS
start_time <- Sys.time()
iss_values <- map_dbl(k.values, iss)
end_time <- Sys.time()
paste("The running time was: ", end_time - start_time, sep=' ')
# Plotting the results
par(mfrow=c(1,1))
plot(k.values,
     iss_values,
     type='b', pch=19, frame=F,
     xlab='Number of Clusters K',
     ylab='Total Intra-clusters sum of squares')

# The best value for the clusters number is the one that is closer to the bend of the 'elbow'

k <- 4

# ------- Average Silhouette Method
# The Silhouette method calculates the average silhouette width and, the higher is number is, the better is the algorithm.

library(cluster)
library(grid)
library(gridExtra)

k2 <- kmeans(customer[, 3:5],
             2,
             iter.max=100,
             nstart=50,
             algorithm='Lloyd')
s2 <- plot(silhouette(k2$cluster, dist(customer[, 3:5], 'euclidean')))

k3 <- kmeans(customer[, 3:5],
             3,
             iter.max=100,
             nstart=50,
             algorithm='Lloyd')
s3 <- plot(silhouette(k3$cluster, dist(customer[, 3:5], 'euclidean')))

k4 <- kmeans(customer[, 3:5],
             4,
             iter.max=100,
             nstart=50,
             algorithm='Lloyd')
s4 <- plot(silhouette(k4$cluster, dist(customer[, 3:5], 'euclidean')))

k5 <- kmeans(customer[, 3:5],
             5,
             iter.max=100,
             nstart=50,
             algorithm='Lloyd')
s5 <- plot(silhouette(k5$cluster, dist(customer[, 3:5], 'euclidean')))

k6 <- kmeans(customer[, 3:5],
             6,
             iter.max=100,
             nstart=50,
             algorithm='Lloyd')
s6 <- plot(silhouette(k6$cluster, dist(customer[, 3:5], 'euclidean')))

k7 <- kmeans(customer[, 3:5],
             7,
             iter.max=100,
             nstart=50,
             algorithm='Lloyd')
s7 <- plot(silhouette(k7$cluster, dist(customer[, 3:5], 'euclidean')))

k8 <- kmeans(customer[, 3:5],
             8,
             iter.max=100,
             nstart=50,
             algorithm='Lloyd')
s8 <- plot(silhouette(k8$cluster, dist(customer[, 3:5], 'euclidean')))

k9 <- kmeans(customer[, 3:5],
             9,
             iter.max=100,
             nstart=50,
             algorithm='Lloyd')
s9 <- plot(silhouette(k9$cluster, dist(customer[, 3:5], 'euclidean')))

k10 <- kmeans(customer[, 3:5],
             10,
             iter.max=100,
             nstart=50,
             algorithm='Lloyd')
s10 <- plot(silhouette(k10$cluster, dist(customer[, 3:5], 'euclidean')))


# Now we make a visualization of the results using the fviz_nbclust() function
# This function helps us visualize the optimal number of clusters as follows

library(NbClust)
library(factoextra)

# Execute the function to get the plot of model performance
fviz_nbclust(customer[, 3:5],
             kmeans,
             method='silhouette')


## ----------- Gap Statistic Method

stat_gap <- clusGap(customer[, 3:5],
                    FUN=kmeans,
                    nstart=25,
                    K.max=10, B=50)
fviz_gap_stat(stat_gap)

# It is seems reasonable to choose K=6, as it has been suggested by 2 methods.
print(k6)

# Printing the model k6 we can see a bunch of information about the model:
# Cluster: A vector that denotes in each cluster each row of the data frame is.
# Centers: A matrix that holds the centroids of each cluster (for each feature[column])
# totss: Total Sum of Squares
# withinss: This is a vector representing the intra-cluster sum of squares having one componenet per cluster
# tot.withinss: This denotes the total intra-cluster sum of squares
# betweens: This is the sum of between-cluster squares
# size: The total number of points that each cluster holds.

# ---- Visualizing the Clustering Results using the First Two Principal Components
# In order to make a 2D visualization, we have to select 2 of the 3 features that our model was built upon.
# For this, we run a Principal Component Analysis to select the 2 most important features.

pcclust=prcomp(customer[,3:5],scale=FALSE) #principal component analysis
summary(pcclust)
pcclust$rotation[,1:2]
# We see that the top 2 most important principal components are the Annual Income and the Spending Score.
# This means that this two features are the ones that most separates the customers.


# Lets visualize the clustering
set.seed(1)

ggplot(customer,
       aes(x=Annual.Income..k..,
           y=Spending.Score..1.100.)) +
  geom_point(stat='identity',
             aes(color=as.factor(k6$cluster))) +
  scale_color_discrete(name=" ",
                       breaks=c('1', '2', '3', '4', '5', '6'),
                       labels=c('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6')) +
  ggtitle("Segments of Mall Customers",
          subtitle='Using K-means Clustering')


# But we can explore the clustering result with the third component, the age, as well.
# Income and Age
ggplot(customer,
       aes(x=Annual.Income..k..,
           y=Age)) +
  geom_point(stat='identity',
             aes(color=as.factor(k6$cluster))) +
  scale_color_discrete(name=" ",
                       breaks=c('1', '2', '3', '4', '5', '6'),
                       labels=c('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6')) +
  ggtitle("Segments of Mall Customers",
          subtitle='Using K-means Clustering')

# As we can see, the age is not a good parameter for segmenting the customers of this dataset!
# Spending score and Age
ggplot(customer,
       aes(x=Spending.Score..1.100.,
           y=Age)) +
  geom_point(stat='identity',
             aes(color=as.factor(k6$cluster))) +
  scale_color_discrete(name=" ",
                       breaks=c('1', '2', '3', '4', '5', '6'),
                       labels=c('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6')) +
  ggtitle("Segments of Mall Customers",
          subtitle='Using K-means Clustering')

# So, the annual income and spending score are the best features to segment our customers.
ggplot(customer,
       aes(x=Annual.Income..k..,
           y=Spending.Score..1.100.)) +
  geom_point(stat='identity',
             aes(color=as.factor(k6$cluster))) +
  scale_color_discrete(name=" ",
                       breaks=c('1', '2', '3', '4', '5', '6'),
                       labels=c('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6')) +
  ggtitle("Segments of Mall Customers",
          subtitle='Using K-means Clustering') 

# Besides k=6 has been the best number of clusters suggested by the Gap Statistic and the Average Silhouette Method, it seems
# That the clusters 1 and 2 are really close to each other, i.e, there should be 5 clusters instead of 6. Lets try making it with 5:

ggplot(customer,
       aes(x=Annual.Income..k..,
           y=Spending.Score..1.100.)) +
  geom_point(stat='identity',
             aes(color=as.factor(k5$cluster))) +
  scale_color_discrete(name=" ",
                       breaks=c('1', '2', '3', '4', '5'),
                       labels=c('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5')) +
  ggtitle("Segments of Mall Customers",
          subtitle='Using K-means Clustering') 
# Now it looks better. But how about the numbers?

k5$tot.withinss
k6$tot.withinss

k5$betweenss
k6$betweenss

# For k6 we have a smaller number for the total sum of ISS than the k5 model. However, k5 presents a smaller number
# of between Iss than the k6. This should be considered when taking this model into production.