#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <igl/readPLY.h>
#include <igl/knn.h>
#include <igl/octree.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/get_seconds.h>
#include "gnuplot_i.h"


using namespace std;
Eigen::MatrixXd V1;
Eigen::MatrixXi F1;
int point_size = 10;
int displaying = 0;
vector<Eigen::MatrixXd> iterations;
vector<double> sqrDiff;

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier){
    if(key == '1'){
        point_size++;
        viewer.data().point_size = point_size;
    }else if(key == '2'){
        if(point_size > 0){
            point_size--;
            viewer.data().point_size = point_size;
        }
    }else if(key == '3'){
        if(displaying != 0){
            displaying--;
            cout << "displaying iteration: " << displaying << endl;
            viewer.data().clear();
            viewer.data().add_points(V1, Eigen::RowVector3d(1, 0, 0.5));
            viewer.data().add_points(iterations[displaying], Eigen::RowVector3d(0.4, 0.5, 0.1));
        }
    }else if(key == '4'){
        if(displaying != (iterations.size()-1)){
            displaying++;
            cout << "displaying iteration: " << displaying << endl;
            viewer.data().clear();
            viewer.data().add_points(V1, Eigen::RowVector3d(1, 0, 0.5));
            viewer.data().add_points(iterations[displaying], Eigen::RowVector3d(0.4, 0.5, 0.1));
        }
    }
    return false;
}

void getSamples(Eigen::MatrixXd& ptSamples, Eigen::MatrixXd& points, int numberOfSamples){
    /*
     * Here we will return a random set of samples and output them into matrix passed
     * through the ptSamples matrix reference
     */

    if(ptSamples.rows() != numberOfSamples){
        cout << "Number of samples does not match output Matrix rows" << endl;
        exit(0);
    }
    if(numberOfSamples == points.rows()){
        //If the number of samples required is the same number as all the points
        //Then we will just copy the points matrix into ptSamples
        ptSamples << points;
        return;
    }
    //TODO::Implement a random sampler and return the sample in ptSamples
    Eigen::VectorXi sampledIndices(numberOfSamples);
    for(int sample = 0; sample < numberOfSamples; sample++){
        int index;
        int unique = 0;
        while(!unique){
            index = rand() % points.rows();
            unique = 1;
            int sampledIndex = 0;
            while(sampledIndex <= sample){
                if(sampledIndices(sampledIndex) == index){
                    unique = 0;
                    break;
                }
                sampledIndex++;
            }
        }
        ptSamples.row(sample) = points.row(index);
    }
}

void computeRt(Eigen::MatrixXd& R, Eigen::RowVector3d& t, Eigen::MatrixXd& sampledPts, Eigen::MatrixXd& sampledPtsNN){
    //Alignment phase
    /*
     * 1. Compute the barycenters for each point sets. There are 2 point sets I have:
     *    The sampled points P and the nn of the sampled points Q
     * 2. Compute the new pointsets P_hat and Q_hat
     * 3. Compute matrix A
     * 4. Compute SVD of matrix A = ULV'
     * 5. Compute R = VU'
     * 6. Compute t = p_barycenter - R*q_barycenter
     */

    //Computing barycenters for samples and its nn
    Eigen::RowVector3d samplesBC = sampledPts.colwise().sum()/sampledPts.rows();
    Eigen::RowVector3d samplesNNBC = sampledPtsNN.colwise().sum()/sampledPtsNN.rows();


    //Compute the recentered points
    Eigen::MatrixXd hatSamples = sampledPts.rowwise() - samplesBC;
    Eigen::MatrixXd hatSamplesNN = sampledPtsNN.rowwise() - samplesNNBC;
    /*
    Eigen::MatrixXd hatSamples = Eigen::MatrixXd::Zero(sampledPts.rows(), sampledPts.cols());
    Eigen::MatrixXd hatSamplesNN = Eigen::MatrixXd::Zero(sampledPtsNN.rows(), sampledPtsNN.cols());

    for(int rowId = 0; rowId < sampledPts.rows(); rowId++){
        hatSamples.row(rowId) = sampledPts.row(rowId) - samplesBC;
        hatSamplesNN.row(rowId) = sampledPtsNN.row(rowId) - samplesNNBC;
    }
    */

    //Assemble matrix A
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
    for(int rowId = 0; rowId < sampledPts.rows(); rowId++){
        A = A + (hatSamples.row(rowId).transpose() * hatSamplesNN.row(rowId));
    }

    //Compute the SVD of A then assemble R and t
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixV() * svd.matrixU().transpose();
    t = samplesNNBC - ((R*samplesBC.transpose()).transpose());


}

double computeSqrDiff(Eigen::MatrixXd& samples, Eigen::MatrixXd& samplesNN){
    double sqrDiff = (samples - samplesNN).cwiseAbs2().sum();
    return sqrDiff;
}

void wait_for_key ()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__TOS_WIN__)  // every keypress registered, also arrow keys
    cout << endl << "Press any key to continue..." << endl;

    FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE));
    _getch();
#elif defined(unix) || defined(__unix) || defined(__unix__) || defined(__APPLE__)
    cout << endl << "Press ENTER to continue..." << endl;

    std::cin.clear();
    std::cin.ignore(std::cin.rdbuf()->in_avail());
    std::cin.get();
#endif
    return;
}

int main(int argc, char* args[]){
    //parameters
    double acceptanceThreshold = 3;
    int iter = 200;
    int numOfSamples = 5000;
    if(argc > 1)
    {
        iter = stoi(args[1]);
        acceptanceThreshold = stoi(args[2]);
        numOfSamples = stoi(args[3]);
    }

    //Load in the files

    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;
    igl::readPLY("../bunny_v2/bun000_v2.ply", V1, F1);
    if(argc > 1){
        igl::readPLY(args[4], V2, F2);
    }else{
        igl::readPLY("../bunny_v2/bun045_v2.ply", V2, F2);
    }



    //Create viewer for displaying the file
    igl::opengl::glfw::Viewer viewer;

    //Construct our octree out of our Q Points (that we are trying to map to)
    //Initialize Octree properties
    std::vector<std::vector<int>> O_PI; //Points inside each Octree cell
    Eigen::MatrixXi O_CH; //Children of each node in the Octree
    Eigen::MatrixXd O_CN; //Mid point of each node in the Octree
    Eigen::VectorXd O_W; //Width of each node in the Octree
    //Compute Octree
    igl::octree(V1, O_PI, O_CH, O_CN, O_W);

    Eigen::MatrixXd currV2 = V2;
    iterations.push_back(V2);

    for(int i = 0; i < iter; i++) {
        //Take samples from V2 (which change every iteration)
        Eigen::MatrixXd samples(numOfSamples, 3);
        getSamples(samples, currV2, numOfSamples);

        //We have our set of points P
        //Next we need to find the closest point for each p in Q

        //Set up results matrix for kNN
        Eigen::MatrixXi kNNResults; //This will hold the kNN for each point we give to the algorithm
        int numNN = 1;
        double tic = igl::get_seconds();
        igl::knn(samples, V1, numNN, O_PI, O_CH, O_CN, O_W, kNNResults);
        printf("%0.4f secs\n",igl::get_seconds()-tic);

        //Collect the V1 points found
        Eigen::MatrixXd samplesNN(samples.rows(), samples.cols());
        for (int sample = 0; sample < numOfSamples; sample++) {
            int rowIndex = kNNResults.row(sample)(0);
            samplesNN.row(sample) = V1.row(rowIndex);
        }
        cout << "passed collecting points" << endl;

        //TODO::Create a outlier rejection phase

        Eigen::MatrixXd sampleDists = (samples - samplesNN).cwiseAbs2().rowwise().sum();
        std::vector<int> keepers;
        double averageDist = sampleDists.sum() / numOfSamples;
        for(int sample=0; sample < numOfSamples; sample++){
            if(sampleDists(sample) < (averageDist * acceptanceThreshold)){
                keepers.push_back(sample);
            }
        }
        Eigen::MatrixXd keptSamples = Eigen::MatrixXd::Zero(keepers.size(), 3);
        Eigen::MatrixXd keptSamplesNN = Eigen::MatrixXd::Zero(keepers.size(), 3);
        for(int keeperIndex = 0; keeperIndex < keepers.size(); keeperIndex++){
            int index = keepers[keeperIndex];
            keptSamples.row(keeperIndex) = samples.row(index);
            keptSamplesNN.row(keeperIndex) = samplesNN.row(index);
        }

        //Alignment phase
        Eigen::MatrixXd R;
        Eigen::RowVector3d t;
        computeRt(R, t, keptSamples, keptSamplesNN);
        currV2 = (R*currV2.transpose()).transpose().rowwise() + t;
        iterations.push_back(currV2);
        cout << "Iteration number: " << i << endl;

        double epsilon = computeSqrDiff(currV2, V1);
        sqrDiff.push_back(epsilon);
        if(i != 0 && abs(sqrDiff[i-1] - epsilon) < 0.001){
            Gnuplot g1("lines");
            //g1.set_title("Convergence");
            g1.plot_x(sqrDiff, "Convergence");
            g1.showonscreen();
            wait_for_key();
            break;
        }
    }
    viewer.data().point_size = point_size;
    viewer.data().add_points(V1, Eigen::RowVector3d(1, 0, 0.5)); //pink
    viewer.data().add_points(currV2, Eigen::RowVector3d(0.4, 0.5, 0.1)); //greenish
    viewer.callback_key_down =  &key_down;
    viewer.launch();

    return 0;
}
