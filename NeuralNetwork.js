function sigmoid(t) {
    return 1/(1+Math.exp(-t));
}
//fake derivative of sigmoid
function dsigmoid(x) {
    return x * (1-x);
}

class NeuralNetwork
{
	constructor(input_num,hidden_num,output_num)
	{
		this.input_num = input_num;
		this.hidden_num = hidden_num;
		this.output_num = output_num;
		
		this.weightsToHidden = new Matrix(this.hidden_num,this.input_num);
		this.weightsToOutput = new Matrix(this.output_num,this.hidden_num);
		this.weightsToHidden.randomize();
		this.weightsToOutput.randomize();
		
		this.biasToHidden = new Matrix(this.hidden_num,1);
		this.biasToOutput = new Matrix(this.output_num,1);
		this.biasToHidden.randomize();
		this.biasToOutput.randomize();
		this.learningRate = 0.1;
		
	}
	
	guess(inputs)
	{
		let inputMatrix = Matrix.fromArray(inputs);
		let hiddenNodes = Matrix.multiply(this.weightsToHidden,inputMatrix);
		hiddenNodes = Matrix.add(hiddenNodes,this.biasToHidden);
		hiddenNodes = Matrix.map(hiddenNodes,sigmoid);
		
		let outputNodes = Matrix.multiply(this.weightsToOutput,hiddenNodes);
		outputNodes = Matrix.add(outputNodes,this.biasToOutput);
		outputNodes = Matrix.map(outputNodes,sigmoid);
		
		let guesses = outputNodes;
		return Matrix.toArray(guesses);
	}
	
	train(inputs,answer)
	{
		let inputMatrix = Matrix.fromArray(inputs);
		let hiddenNodes = Matrix.multiply(this.weightsToHidden,inputMatrix);
		hiddenNodes = Matrix.add(hiddenNodes,this.biasToHidden);
		hiddenNodes = Matrix.map(hiddenNodes,sigmoid);
		
		let outputNodes = Matrix.multiply(this.weightsToOutput,hiddenNodes);
		outputNodes = Matrix.add(outputNodes,this.biasToOutput);
		outputNodes = Matrix.map(outputNodes,sigmoid);
		
		let guesses = outputNodes;
		let answers = Matrix.fromArray(answer);
		let output_errors = Matrix.subtract(answers,guesses);
		
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
	 mutate(func) {
	    this.weights_ih.map(func);
	    this.weights_ho.map(func);
	    this.bias_h.map(func);
	    this.bias_o.map(func);
 	}
	
}	
