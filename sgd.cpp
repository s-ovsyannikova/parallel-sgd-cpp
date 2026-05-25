#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <iomanip>

using namespace std;

/*
 MSE loss
*/
double calculate_loss(const vector<vector<double>>& X, const vector<double>& y, const vector<double>& weights) {
    double loss = 0.0;
    int N = X.size();
    if (N == 0) return 0.0;
    
    #pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < N; ++i) {
        double prediction = 0.0;
        for (size_t j = 0; j < weights.size(); ++j) {
            prediction += X[i][j] * weights[j];
        }
        loss += pow(prediction - y[i], 2);
    }
    return loss / N;
}

/*
 Generate synthetic data for linear regression
*/

void generate_data(int N, int P, vector<vector<double>>& X, vector<double>& y, const vector<double>& true_weights) {
    X.resize(N, vector<double>(P));
    y.resize(N);
    
    random_device rd;
    mt19937 generator(rd());
    normal_distribution<> distribution(0.0, 1.0);		// features
    uniform_real_distribution<> noise_dist(-5.0, 5.0);	// noise

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            X[i][j] = distribution(generator);
        }

        // target: y = X * w_true + f_non_linear(X) + noise
        double true_value = 0.0;
        
        for (int j = 0; j < P; ++j) {
            true_value += X[i][j] * true_weights[j];
        }
        
        true_value += 5.0 * sin(X[i][0]);
        
        y[i] = true_value + noise_dist(generator); 
    }
}

/*
 Save loss to .csv
*/
void save_loss_history(const vector<double>& loss_history, const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        file << "Epoch,Loss\n";
        for (size_t i = 0; i < loss_history.size(); ++i) {
            file << i + 1 << "," << loss_history[i] << "\n";
        }
        file.close();
        cout << "Loss history is saved to file: " << filename << endl;
    } else {
        cerr << "Couldn't open file: " << filename << endl;
    }
}

/*
 Sequential SGD
*/

void sequential_sgd(
    const vector<vector<double>>& X,
    const vector<double>& y,
    vector<double>& weights,
    double learning_rate,
    int epochs,
    int batch_size,
    vector<double>& loss_history
) {
    int N = X.size();
    int P = weights.size();
    int num_batches = N / batch_size;

    random_device rd;
    mt19937 g(rd());
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        shuffle(indices.begin(), indices.end(), g);

        for (int b = 0; b < num_batches; ++b) {
            vector<double> gradient(P, 0.0);

            // calculate gradient for mini-batch
            for (int k = 0; k < batch_size; ++k) {
                int data_idx = indices[b * batch_size + k];
                double prediction = 0.0;

                for (int j = 0; j < P; ++j) {
                    prediction += X[data_idx][j] * weights[j];
                }

                double error = prediction - y[data_idx];

                for (int j = 0; j < P; ++j) {
                    gradient[j] += error * X[data_idx][j];
                }
            }

            // weights update
            for (int j = 0; j < P; ++j) {
                weights[j] -= learning_rate * (gradient[j] / batch_size);
            }
        }
        loss_history.push_back(calculate_loss(X, y, weights));
    }
}

/*
 Parallel SGD
*/

void parallel_sgd(
    const vector<vector<double>>& X,
    const vector<double>& y,
    vector<double>& weights,
    double learning_rate,
    int epochs,
    int batch_size,
    vector<double>& loss_history
) {
    int N = X.size();
    int P = weights.size();
    int num_batches = N / batch_size;

    random_device rd;
    mt19937 g(rd());
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        shuffle(indices.begin(), indices.end(), g);

        for (int b = 0; b < num_batches; ++b) {
            vector<double> gradient(P, 0.0);

			// Parallel gradient calculation for mini-batch objects
            #pragma omp parallel
            {
                vector<double> local_gradient(P, 0.0);
                
                #pragma omp for nowait
                for (int k = 0; k < batch_size; ++k) {
                    int data_idx = indices[b * batch_size + k];
                    double prediction = 0.0;

                    for (int j = 0; j < P; ++j) {
                        prediction += X[data_idx][j] * weights[j];
                    }

                    double error = prediction - y[data_idx];

                    for (int j = 0; j < P; ++j) {
                        local_gradient[j] += error * X[data_idx][j];
                    }
                }
                
				// aggregate local gradients
                #pragma omp critical
                {
                    for (int j = 0; j < P; ++j) {
                        gradient[j] += local_gradient[j];
                    }
                }
            }

            // update weights (sequentially)
            for (int j = 0; j < P; ++j) {
                weights[j] -= learning_rate * (gradient[j] / batch_size);
            }
        }
        loss_history.push_back(calculate_loss(X, y, weights));
    }
}

/*
 Testing
*/

int main(int argc, char* argv[]) {
    if (argc < 2){
        cerr << "Trying to use " << argv[0] << "<batch_size>" << endl;
        return 1;
    }
    // model and data parameters
    int N = 200000;         // number of data objects
    int P = 20;             // number of features
    int epochs = 30;        // number of epochs
    double learning_rate = 0.005;

    int batch_size = std::atoi(argv[1]);
    if (batch_size <=0 || batch_size > N) {
        cerr << "Incorrect batch size: " << batch_size << endl;
        return 1;
    }

    cout << "Running with batch_size=" << batch_size << "and " << omp_get_max_threads() << " threads." << endl; 

    vector<double> true_weights(P);
    for (int j = 0; j < P; ++j) true_weights[j] = 1.0; 

    // data generating
    vector<vector<double>> X;
    vector<double> y;
    cout << "Generating data (N=" << N << ", P=" << P << ")" << endl;
    generate_data(N, P, X, y, true_weights);

    // TESTING SEQUENTIAL SGD
    cout << "\n--- Sequential SGD ---" << endl;
    vector<double> seq_weights(P, 0.1); 
    vector<double> seq_loss_history;

    auto start_seq = chrono::high_resolution_clock::now();
    sequential_sgd(X, y, seq_weights, learning_rate, epochs, batch_size, seq_loss_history);
    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_seq = end_seq - start_seq;

    cout << "Computation time for sequential: " << fixed << setprecision(4) << duration_seq.count() << "s" << endl;
    cout << "Final loss for sequential (MSE): " << seq_loss_history.back() << endl;
    string seq_filename = "losses/loss_seq_b" + to_string(batch_size) + ".csv"; 
    save_loss_history(seq_loss_history, seq_filename);
    
    // TESTING PARALLEL SGD
    cout << "\n--- Parallel SGD ---" << endl;
    
    int num_threads = omp_get_max_threads();
    cout << "Using threads: " << num_threads << endl;

    vector<double> par_weights(P, 0.1);
    vector<double> par_loss_history;
    
    auto start_par = chrono::high_resolution_clock::now();
    parallel_sgd(X, y, par_weights, learning_rate, epochs, batch_size, par_loss_history);
    auto end_par = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_par = end_par - start_par;

    cout << "Computation time for parallel: " << fixed << setprecision(4) << duration_par.count() << "s" << endl;
    cout << "Final loss for parallel (MSE): " << par_loss_history.back() << endl;

    string par_filename = "losses/loss_par_b" + to_string(batch_size) + "_p" + to_string(omp_get_max_threads()) + ".csv"; 
    save_loss_history(par_loss_history, par_filename);

    // Analysis
    double speedup = duration_seq.count() / duration_par.count();
    cout << "\n--- Analysis ---" << endl;
    cout << "Speedup with " << num_threads << " threads: " << fixed << setprecision(2) << speedup << endl;

    return 0;
}
