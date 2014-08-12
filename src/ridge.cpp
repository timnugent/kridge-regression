/*

Ridge regression using Eigen by Tim Nugent 2014

*/

#include <fstream>
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class Ridge{

public:
	Ridge() : loaded(0), kernel_type(1), normalise(0), lambda(0.001), gamma(0.001), constant(1.0), order(2.0) {}
	Ridge(const char* data, char sep = ',') : loaded(0), kernel_type(1), normalise(0), lambda(0.001), gamma(0.001), constant(1.0), order(2.0)  {loaded = load_data(data, sep);}
	int load_data(const char* data, char sep);
	void set_lambda(double l){lambda = l;}
	void ols(){lambda = 0.0; ridge();}
	void ridge();
	double kridge(VectorXd&);
	void print();
	void set_kernel(const unsigned int i){kernel_type = i;};	
	void set_normalise(const unsigned int i){normalise = i;};
	void set_gamma(const double i){gamma = i;};
	void set_constant(const double i){constant = i;};
	void set_order(const double i){order = i;};
private:
	double kernel(const VectorXd& a, const VectorXd& b);
	void generate_kernel_matrix();
	MatrixXd X, K;
	VectorXd Y, B, M, Xscale, k;
	unsigned int loaded, kernel_type, normalise;
	double lambda, ymean, gamma, constant, order;

};

int Ridge::load_data(const char* data, char sep){

	// Read data
	unsigned int row = 0;
	ifstream file(data);
	if(file.is_open()){
		string line,token;
		while(getline(file, line)){
			stringstream tmp(line);
			unsigned int col = 0;
			while(getline(tmp, token, sep)){
				if(col == 0){
					Y.conservativeResize(row+1);
					Y(row) = atof(token.c_str());
				}else{
					if(!row){
						X.conservativeResize(row+1,col);
					}else{
						X.conservativeResize(row+1,X.cols());
					}
					X(row,col-1) = atof(token.c_str());
				}
				col++;
			}
			row++;
		}
		file.close();
	}else{
		cout << "Failed to read file " << data << endl;
		return(0);
	}
	B.resize(Y.rows());

	// Mean centre Y's
	ymean = Y.colwise().mean()(0);
	Y.array() -= ymean;

	// Calculate X means
	M = X.colwise().mean();

	// Calculate X's population standard deviations
	Xscale = ArrayXd::Zero(X.cols());
	for(unsigned int i = 0; i < X.rows(); i++){
		for(unsigned int j = 0; j < X.cols(); j++){
			Xscale(j) += ((M(j)-X(i,j))*(M(j)-X(i,j)));
		}	
	}
	for(unsigned int j = 0; j < X.cols(); j++){
		Xscale(j) = sqrt(Xscale(j)/(double)X.rows());
	}	

	// Mean centre X's and divide by population standard deviations
	for(unsigned int i = 0; i < X.rows(); i++){
		for(unsigned int j = 0; j < X.cols(); j++){
			X(i,j) -= M(j);
			X(i,j) /= Xscale(j);
		}	
	}

	loaded = 1;
	return(1);
}

void Ridge::ridge(){

	if(loaded){
		MatrixXd A = X.transpose() * X;
		MatrixXd R = MatrixXd::Identity(A.rows(),A.cols());
		R *= lambda;
		A += R;
		B = A.inverse()*X.transpose()*Y;
	}
}

double Ridge::kridge(VectorXd& x_h){

	if(loaded){

		generate_kernel_matrix();
		k.resize(Y.rows());
		for(unsigned int i = 0; i < k.rows(); i++){
			k(i) = kernel(X.row(i),x_h);
		}
		MatrixXd R = MatrixXd::Identity(K.rows(),K.cols());
		R *= lambda;
		K += R;

		double y_h = Y.transpose()*K.inverse()*k; 
		return(y_h);

	}else{
		cout << "No data loaded!" << endl;
		return(0);
	}
}

double Ridge::kernel(const VectorXd& a, const VectorXd& b){

	switch(kernel_type){
	    case 2  :
	    	return(pow(a.dot(b)+constant,order));
	    default : 
	    	return(exp(-gamma*((a-b).squaredNorm())));
	}

}

void Ridge::generate_kernel_matrix(){

	// Fill kernel matrix
	K.resize(X.rows(),X.rows());
	for(unsigned int i = 0; i < X.rows(); i++){
		for(unsigned int j = i; j < X.rows(); j++){
			K(i,j) = K(j,i) = kernel(X.row(i),X.row(j));
		}	
	}

	cout << K << endl;

	// Normalise kernel matrix	
	if(normalise){
		VectorXd d = K.rowwise().sum();
		for(unsigned int i = 0; i < d.rows(); i++){
			d(i) = 1.0/sqrt(d(i));
		}
		auto F = d.asDiagonal();
		MatrixXd l = (K * F);
		for(unsigned int i = 0; i < l.rows(); i++){
			for(unsigned int j = 0; j < l.cols(); j++){
				l(i,j) = l(i,j) * d(i);
			}
		}		
		K = l;

		cout << endl << "Normalised kernel matrix:" << endl;
		cout << K << endl;
	}

}

void Ridge::print(){

	if(loaded){
		double intercept = ymean;
		for(unsigned int i = 0; i < B.rows(); i++){
			intercept -= (B(i)/Xscale(i))*M(i);
		}
		printf(" %2.6f",intercept);
		for(unsigned int i = 0; i < B.rows(); i++){
			printf(" %2.6f",B(i)/Xscale(i));
		}
		cout << endl;
	}
}

int main(int argc, const char* argv[]){

	if(argc < 2){
		cout << "Usage:\n" << argv[0] << " <DATA>" << endl;
		cout << "File format:\nY,X1,X2, ... Xn\n";
		return(0);
	}
	Ridge* R = new Ridge(argv[1],',');
	for(double l = 0.0; l <= 1.0; l += 0.001){
		printf("%1.3f",l);
		R->set_lambda(l);
		R->ridge();
		R->print();
	}


	cout << endl << "Kernel Ridge Regression:" << endl;
	R->set_normalise(0);
	VectorXd x_h(2);
	x_h(0) = 5.2;
	x_h(1) = 1.9;
	double y_h = R->kridge(x_h);
	cout << "x' = " << endl << x_h << endl;
	cout << "y' = " << y_h << endl;
	delete R;
	return(0);

}

