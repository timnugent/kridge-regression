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
	Ridge() : lambda(0.001), loaded(0) {}
	Ridge(const char* data, char sep = ',') : lambda(0.001), loaded(0) {loaded = load_data(data, sep);}
	int load_data(const char* data, char sep);
	void set_lambda(double l){lambda = l;}
	void ols(){lambda = 0.0; ridge();}
	void ridge();
	void print();

private:
	MatrixXd X;
	VectorXd Y, B, M, Xscale;
	double lambda, ymean;
	int loaded;

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
	for(unsigned int i = 0; i < Y.rows(); i++){
		Y(i) -= ymean;
	}

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
		MatrixXd A = X.adjoint() * X;
		MatrixXd R = MatrixXd::Identity(A.rows(),A.cols());
		R *= lambda;
		A += R;
		B = A.inverse()*X.adjoint()*Y;
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
	delete R;
	return(0);

}

