function sigmoid(t) {
    return 1/(1+Math.exp(-t));
}
//fake derivative of sigmoid
function dsigmoid(x) {
    return x * (1-x);
}

class NeuralNetwork
{
	constructor(input_num,hidden_num,output_num,hiddenLayers_num)
	{
		this.input_num = input_num;
		this.hidden_num = hidden_num;
		this.output_num = output_num;
		this.hiddenLayers_num = hiddenLayers_num;
		this.AdditionalHiddenLayers = [];
		
		this.weightsToHidden = new Matrix(this.hidden_num,this.input_num);
		this.weightsToOutput = new Matrix(this.output_num,this.hidden_num);
		this.weightsToHidden.randomize();
		this.weightsToOutput.randomize();
		
		this.biasToHidden = new Matrix(this.hidden_num,1);
		this.biasToOutput = new Matrix(this.output_num,1);
		this.biasToHidden.randomize();
		this.biasToOutput.randomize();
		this.learningRate = 0.1;
		
		for(var i =1;i<hiddenLayers_num;i++)
		{
			var w = new Matrix(this.hidden_num,this.hidden_num);
			w.randomize();
			var b = new Matrix(this.hidden_num,1);
			b.randomize();
			this.AdditionalHiddenLayers.push(new HiddenLayer(w,b));
		}
		
	}
	
	guess(inputs)
	{
		let inputMatrix = Matrix.fromArray(inputs);
		let hiddenNodes = Matrix.multiply(this.weightsToHidden,inputMatrix);
		hiddenNodes = Matrix.add(hiddenNodes,this.biasToHidden);
		hiddenNodes = Matrix.map(hiddenNodes,sigmoid);
		var lastHidden = hiddenNodes;
		if(this.hiddenLayers_num>1)
		{
			let addHiddenNodes = Matrix.multiply(this.AdditionalHiddenLayers[0].weights,hiddenNodes);
			addHiddenNodes = Matrix.add(addHiddenNodes,this.AdditionalHiddenLayers[0].biases);
			addHiddenNodes = Matrix.map(addHiddenNodes,sigmoid);
			lastHidden = addHiddenNodes;
			for(var i =0;i<this.AdditionalHiddenLayers.length;i++)
			{
				var temp = Matrix.multiply(this.AdditionalHiddenLayers[i].weights,lastHidden);
				temp = Matrix.add(temp,this.AdditionalHiddenLayers[i].biases);
				temp = Matrix.map(temp,sigmoid);
				lastHidden = temp;
				this.AdditionalHiddenLayers[i].nodes = temp;
			}
		}
		
		
		let outputNodes = Matrix.multiply(this.weightsToOutput,lastHidden);
		outputNodes = Matrix.add(outputNodes,this.biasToOutput);
		outputNodes = Matrix.map(outputNodes,sigmoid);
		
		let guesses = outputNodes;
		return Matrix.toArray(guesses);
	}
	
	train(inputs,answer)
	{
		//guess and get the output errors
		let inputMatrix = Matrix.fromArray(inputs);
		let hiddenNodes = Matrix.multiply(this.weightsToHidden,inputMatrix);
		hiddenNodes = Matrix.add(hiddenNodes,this.biasToHidden);
		hiddenNodes = Matrix.map(hiddenNodes,sigmoid);
		var lastHidden = hiddenNodes;
		if(this.hiddenLayers_num>1)
		{
			let addHiddenNodes = Matrix.multiply(this.AdditionalHiddenLayers[0].weights,hiddenNodes);
			addHiddenNodes = Matrix.add(addHiddenNodes,this.AdditionalHiddenLayers[0].biases);
			addHiddenNodes = Matrix.map(addHiddenNodes,sigmoid);
			lastHidden = addHiddenNodes;
			for(var i =0;i<this.AdditionalHiddenLayers.length;i++)
			{
				var temp = Matrix.multiply(this.AdditionalHiddenLayers[i].weights,lastHidden);
				temp = Matrix.add(temp,this.AdditionalHiddenLayers[i].biases);
				temp = Matrix.map(temp,sigmoid);
				lastHidden = temp;
				this.AdditionalHiddenLayers[i].nodes = temp;
			}
		}
		
		
		let outputNodes = Matrix.multiply(this.weightsToOutput,lastHidden);
		outputNodes = Matrix.add(outputNodes,this.biasToOutput);
		outputNodes = Matrix.map(outputNodes,sigmoid);
		let guesses = outputNodes;
		let answers = Matrix.fromArray(answer);
		let output_errors = Matrix.subtract(answers,guesses);
		
		
		//back propagation for a single hidden layer network
		if(this.hiddenLayers_num==1)
		{
			let tWeightsToOutput = Matrix.transpose(this.weightsToOutput);
			let hiddenErrors = Matrix.multiply(tWeightsToOutput,output_errors);

			////adjust the weights which are heading to the outputs layer
			let gradients = Matrix.map(outputNodes,dsigmoid);
			
			//not a product of two matrices but just element-wise multiplication (hadamard product) the other multiplication is weighted sum
			gradients = Matrix.EWMultiply(gradients,output_errors);
			gradients = Matrix.multiply(gradients,this.learningRate);
			
			let tHidden = Matrix.transpose(hiddenNodes);
			let weightsToOutputDeltas = Matrix.multiply(gradients,tHidden);
			
			//weights
			this.weightsToOutput = Matrix.add(this.weightsToOutput,weightsToOutputDeltas);
			
			//biases (delta for bias is just the gradient)
			this.biasToOutput = Matrix.add(this.biasToOutput,gradients);
				//adjust the weights which are heading to the hidden layer
			let hiddenGradients = Matrix.map(hiddenNodes,dsigmoid);
			
			hiddenGradients = Matrix.EWMultiply(hiddenGradients,hiddenErrors);
			hiddenGradients = Matrix.multiply(hiddenGradients,this.learningRate);
			
			
			let tInput = Matrix.transpose(inputMatrix);
			let weightsToHiddenDeltas = Matrix.multiply(hiddenGradients,tInput);
			
			//weights
			this.weightsToHidden = Matrix.add(this.weightsToHidden,weightsToHiddenDeltas);
			
			//biases
			this.biasToHidden = Matrix.add(this.biasToHidden,hiddenGradients);
		}
		
		//back propagation for a multi hidden layer network
		if(this.hiddenLayers_num>1)
		{
			
			//last hidden layer
			var lastHiddenLayer = this.AdditionalHiddenLayers[this.AdditionalHiddenLayers.length-1];
			var tWeightsToOutput = Matrix.transpose(this.weightsToOutput);
			var errors = Matrix.multiply(tWeightsToOutput,output_errors); 
			lastHiddenLayer.errors = errors;
			
			for(var b=this.AdditionalHiddenLayers.length-2;b>=0;b--)
			{
				var temp = this.AdditionalHiddenLayers[b];
				var tw = Matrix.transpose(lastHiddenLayer.weights);
				var e = Matrix.multiply(tw,lastHiddenLayer.errors);
				this.AdditionalHiddenLayers[b].errors = e;
				lastHiddenLayer = this.AdditionalHiddenLayers[b];
			}
			//first hidden errors
			var firstHiddenErrors;
			var tlhw = Matrix.transpose(lastHiddenLayer.weights);
			var er = Matrix.multiply(tlhw,lastHiddenLayer.errors);
			firstHiddenErrors = er;
			
			let gradients = Matrix.map(outputNodes,dsigmoid);

			gradients = Matrix.EWMultiply(gradients,output_errors);
			gradients = Matrix.multiply(gradients,this.learningRate);
			
			var tlastHiddenNodes = Matrix.transpose(lastHiddenLayer.nodes);
			let weightsToOutputDeltas = Matrix.multiply(gradients,tlastHiddenNodes);
			this.weightsToOutput = Matrix.add(this.weightsToOutput,weightsToOutputDeltas);
			this.biasToOutput = Matrix.add(this.biasToOutput,gradients);
			
			
			for(var m=this.AdditionalHiddenLayers.length-2;b>0;b--)
			{
				var current = this.AdditionalHiddenLayers[b];
				var gradient = Matrix.map(current.nodes,dsigmoid);
				gradient= Matrix.EWMultiply(gradient,current.errors);
				gradient = Matrix.multiply(gradient,this.learningRate);
				
				var transposedNodes = Matrix.transpose(current.nodes);
				var weightsDeltas = Matrix.multiply(gradient,transposedNodes);
				current.weights = Matrix.add(current.weights,weightsDeltas);
				current.biases = Matrix.add(current.biases,gradient);
				this.AdditionalHiddenLayers[b] = current;
			}
			
			
			var firstAdditionalLayer = this.AdditionalHiddenLayers[0];
			var falg = Matrix.map(firstAdditionalLayer.nodes,dsigmoid);
			falg = Matrix.EWMultiply(falg,firstAdditionalLayer.errors);
			falg = Matrix.multiply(falg,this.learningRate);

			let tH = Matrix.transpose(hiddenNodes);
			let weightsToFirstAdditionalHiddenDeltas = Matrix.multiply(falg,tH);
			firstAdditionalLayer.weights = Matrix.add(firstAdditionalLayer.weights,weightsToFirstAdditionalHiddenDeltas);
			firstAdditionalLayer.biases = Matrix.add(firstAdditionalLayer.biases,falg);
			
			this.AdditionalHiddenLayers[0].firstAdditionalLayer;
			
			//adjust the weights which are heading to the hidden layer
			let firstHiddenGradients = Matrix.map(hiddenNodes,dsigmoid);
			
			firstHiddenGradients = Matrix.EWMultiply(firstHiddenGradients,firstHiddenErrors);
			firstHiddenGradients = Matrix.multiply(firstHiddenGradients,this.learningRate);
			
			
			let tInput = Matrix.transpose(inputMatrix);
			let weightsToFirstHiddenDeltas = Matrix.multiply(firstHiddenGradients,tInput);
			
			//weights
			this.weightsToHidden = Matrix.add(this.weightsToHidden,weightsToFirstHiddenDeltas);
			
			//biases
			this.biasToHidden = Matrix.add(this.biasToHidden,firstHiddenGradients);
			
		}
		
		
		
		
	}
	mutate(func) 
	{
	    this.weightsToHidden = Matrix.map(this.weightsToHidden,func);
	    this.weightsToOutput = Matrix.map(this.weightsToOutput,func);
	    this.biasToHidden = Matrix.map(this.biasToHidden,func);
	    this.biasToOutput = Matrix.map(this.biasToOutput,func);
		for(var i in this.AdditionalHiddenLayers)
		{
			this.AdditionalHiddenLayers[i].weights = Matrix.map(this.AdditionalHiddenLayers[i].weights,func);
			this.AdditionalHiddenLayers[i].biases = Matrix.map(this.AdditionalHiddenLayers[i].biases,func);
			
		}
		
 	}
}	

class HiddenLayer
{
	constructor(weights,biases)
	{
		this.weights = weights;
		this.biases = biases;
		this.nodes;
		this.errors;
		this.gradient;
	}
}
