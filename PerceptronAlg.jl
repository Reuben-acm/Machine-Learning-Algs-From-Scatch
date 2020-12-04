### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ cd4985f4-35e7-11eb-1e72-dbe7ddc69f7d
begin
	using Pkg
	using RDatasets
	using Plots
	Pkg.build("GR")
end

# ╔═╡ 6ae26ea4-35eb-11eb-365b-0b46da7b5069
md"# Perceptron ALgorithm"

# ╔═╡ 8957a536-35eb-11eb-184b-bde9b5854440
md" **In this notebook we will be using the Perceptron Algorithm to classify data from the famous [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)**

The packages we will be using for this notebook are:
[Plots](http://docs.juliaplots.org/latest/)
[RDatasets](https://github.com/JuliaStats/RDatasets.jl)
"

# ╔═╡ e717312c-35ec-11eb-335a-47c7e9730aad
	md"In machine learning, the perceptron is an algorithm for supervised learning of binary classifiers. A binary classifier is a function which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.This algorithm is often refered to as the ''Hello World'' of machine learning

Note**:There are two types of Perceptrons:Single Layer & Multilayer. Single layer perceptrons are only capable of learning linearly separable patterns.

![](/Users/rl10/Desktop/perceptron.png)

"

# ╔═╡ f01285c4-35ec-11eb-0d97-c7c18f509d15
md"The goal of this notebook will be to determine which features will the perceptron work best on. We start by visualizing the data to view which set of features look the most  linearly seperable "

# ╔═╡ 1be9e442-35e8-11eb-03e8-43b1a1a872b9
begin
	iris = dataset("datasets","iris")
	first(iris,75)
end

# ╔═╡ be664d62-35ea-11eb-1f57-77c22b0dc69b
begin
	p1 = scatter(iris.PetalWidth[1:100],iris.PetalLength[1:100], xlabel = "PetalWidth", ylabel = "PetalLength",color = "orange")
	p2 = scatter(iris.SepalWidth[1:100],iris.PetalWidth[1:100],xlabel = "SepalWidth", ylabel = "PetalWidth")
	p3 = scatter(iris.SepalLength[1:100],iris.PetalWidth[1:100],xlabel = "SepalLength", ylabel = "PetalWidth",color = "red")
	p4 = scatter(iris.SepalWidth[1:100],iris.PetalLength[1:100],xlabel = "SepalWidth", ylabel = "PetalLength",color = "green")
	plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
end

# ╔═╡ 8d16ba94-35ef-11eb-24aa-2313da868608
md"Through visulization we can see some set of features look more easliy linearly seperable than others"

# ╔═╡ 4fd97812-35f2-11eb-39cf-5713c7ae4561
md"##### Implementing the Perceptron Learning Algorithm "

# ╔═╡ 7d0da66e-35f2-11eb-11ae-458602978760
begin
	
"Extract the first two features from the first 100 rows of data. These two features can be changed to select new features"

x_data = [x for x in zip(iris.PetalWidth[1:100], iris.PetalLength[1:100])]

"Encode labels with numeric value"
y_data = [iris.Species[i] == "setosa" ? 1 : -1 for i = 1:100]


"Plot data to verify the data seperation"

scatter(x_data[1:50], 
        label = "setosa",
        xaxis = "Petal Width",
        yaxis = "Petal Length",
        title = "Iris")


scatter!(x_data[51:100],
        label = "versicolor")
end

# ╔═╡ fce3c54c-35f4-11eb-1e79-b10f2a4fd837
begin
	function Sign(weights, x)
	    """ Sign function input factor in out algorithm. We are using this function to find the 
	        dot product of the weights and x vector in out algorithm.
	    
	    Args:
	        weight (Array): The parameter within a neural network that transforms input data within the network's hidden layers.
	                        These are random input based on the size of your x_data vector size
	    
	        x ([tuple]):    Tuple input of one tuple of the x_data that had been selected before
	    
	    Output:
	                        Returns a value of -1 or 1 depending on the dot product
	    """
	
	    x̂ = [x[1],x[2],1.0]
	    return weights'x̂ > 0 ? 1 : -1
	end
	
	
	function perceptron_update(weights, x,y)
	    """ Function to check output of the sign function to match the y_data. 
	        This update function will only update on error of Sign function and y_data.
	        Otherwise, no update in the weight value
	    
	    Args:
	        weights (Array):Array of updated weight values to be used in network classification
	        x (Tuple):      Tuple input of the data selected one point at a time
	        y (Array) :     Array of listed of -1 or 1 that represents the correct classification of data
	    
	    Output:
	                        Returns the updated or non-updated weight values
	    """
	    if Sign(weights,x) != y
	        weights += y*[x[1],x[2],1.0]
	    end
	    return weights
	end
	
	function Error(weights, feature_set, labels)
	    """ Function that sums an array for the length of the feature set to calcuate the 
	        total amount of errors in the feature set. 0 if correct and 1 if wrong classification 
	    
	    Args:
	        weights (Array):Array of updated weight values to be used in network classification
	        x (Tuple):      Tuple input of the data selected one point at a time
	        y (Array) :     Array of listed of -1 or 1 that represents the correct classifcation of data
	    
	    Output:
	                        Number of errors in this feature set input into the algorithm on this run
	    """
	        return sum([Sign(weights, feature_set[i]) != labels[i] ? 1 : 0 for i = 1:length(feature_set)])
	end
end

# ╔═╡ fd847bfe-35f4-11eb-39ec-4d9f0dd8279c
function perceptron_learning_algorithm(weights, feature_set, labels, ϵ)
    """ Function that uses all the previous created functions. Perceptron Algorithm. 
    
    Args:
        weights (Array):     Array of updated weight values to be used in network classification
        feature set (Tuple): Input of all the x_data selected out before    
        labels (Array):      Array of listed of -1 or 1 that represents the correct classifcation of data based on our data
        ϵ (int):             Epslion is the error value that we target for, function will continue until this error value is reached
    
    Output:
        weights: updated     Weights in the algorithm 
        Weight Vector:       Listed of updated weight values over each iteration 
        Error measures:      Array of the count of errors each iteration 
    """

    error_measures = []
    append!(error_measures, Error(weights,feature_set,labels))
    
    weight_vector = []
    
    while Error(weights, feature_set, labels) > ϵ
        for i = 1:length(feature_set)
            weights = perceptron_update(weights, feature_set[i], labels[i])
        end
        append!(weight_vector,weights)
        append!(error_measures, Error(weights, feature_set, labels))
    end
    
    
    return weights, weight_vector, error_measures
end


# ╔═╡ fe0a30d4-35f4-11eb-19e5-a3a79d5b651e
w, W, errors = perceptron_learning_algorithm(randn(3), x_data, y_data, 1)


# ╔═╡ 9fd575f6-35f6-11eb-31c4-9fa20aac91fa
begin
	plot(errors,label = "Number of misclasifications")
	scatter!(errors)
end

# ╔═╡ 1f341d22-35f5-11eb-1f45-9b26da537717
begin
	scatter(x_data[1:50], 
	        label = "setosa",
	        xaxis = "Petal Width",
	        yaxis = "Petal Length",
	        title = "Iris")
	
	
	scatter!(x_data[51:100],
	        label = "versicolor")
	
	
	plot!(x-> (-x*w[1] - w[3])/w[2])
end

# ╔═╡ 1fc8acda-35f5-11eb-2ef7-2f33cb633958
begin
	md"## Conclusion
	  In this notebook we: 
		* Introduced the Perceptron
		* Visualized the Iris dataset to select the best feature-set that was linearly seperable
		* Implemeted the PLA on selected feature-set achieving minimal error"
end

# ╔═╡ ce587166-35fc-11eb-0924-379d8f6645c2


# ╔═╡ Cell order:
# ╟─6ae26ea4-35eb-11eb-365b-0b46da7b5069
# ╟─8957a536-35eb-11eb-184b-bde9b5854440
# ╟─e717312c-35ec-11eb-335a-47c7e9730aad
# ╟─f01285c4-35ec-11eb-0d97-c7c18f509d15
# ╠═cd4985f4-35e7-11eb-1e72-dbe7ddc69f7d
# ╠═1be9e442-35e8-11eb-03e8-43b1a1a872b9
# ╠═be664d62-35ea-11eb-1f57-77c22b0dc69b
# ╟─8d16ba94-35ef-11eb-24aa-2313da868608
# ╟─4fd97812-35f2-11eb-39cf-5713c7ae4561
# ╠═7d0da66e-35f2-11eb-11ae-458602978760
# ╠═fce3c54c-35f4-11eb-1e79-b10f2a4fd837
# ╠═fd847bfe-35f4-11eb-39ec-4d9f0dd8279c
# ╠═fe0a30d4-35f4-11eb-19e5-a3a79d5b651e
# ╠═9fd575f6-35f6-11eb-31c4-9fa20aac91fa
# ╠═1f341d22-35f5-11eb-1f45-9b26da537717
# ╠═1fc8acda-35f5-11eb-2ef7-2f33cb633958
# ╠═ce587166-35fc-11eb-0924-379d8f6645c2
