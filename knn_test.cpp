#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <igl/readPLY.h>
#include <igl/knn.h>
#include <igl/octree.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/get_seconds.h>


using namespace std;
Eigen::MatrixXd V1;
Eigen::MatrixXi F1;
int point_size = 10;
int displaying = 0;
Eigen::MatrixXd samplesG;
Eigen::MatrixXd samplesNNG;

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

            viewer.data().add_points(samplesG.row(displaying), Eigen::RowVector3d(1, 1, 1));
            viewer.data().add_points(samplesNNG.row(displaying), Eigen::RowVector3d(0, 0, 0));
        }
    }else if(key == '4'){
        if(displaying != samplesG.rows()-1){
            displaying++;
            cout << "displaying iteration: " << displaying << endl;
            viewer.data().clear();
            viewer.data().add_points(samplesG.row(displaying), Eigen::RowVector3d(1, 1, 1));
            viewer.data().add_points(samplesNNG.row(displaying), Eigen::RowVector3d(00, 0, 0));
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
    cout << "samplesBC" << samplesBC << endl;
    cout << "samplesNNBC" << samplesNNBC << endl;


    //Compute the recentered points
    Eigen::MatrixXd hatSamples = sampledPts.rowwise() - samplesBC;
    Eigen::MatrixXd hatSamplesNN = samplesNNBC.rowwise() - samplesNNBC;


    //Assemble matrix A
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
    for(int rowId = 0; rowId < sampledPts.rows(); rowId++){
        A = A + (hatSamples.row(rowId).transpose() * hatSamplesNN.row(rowId));
    }

    //Compute the SVD of A then assemble R and t
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixV() * svd.matrixU().transpose();
    cout << R << endl;
    t = samplesNNBC - (R * samplesBC.transpose()).transpose();

}

int main(int argc, char* args[]){
    //Load in the files

    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;
    igl::readPLY("../bunny_v2/bun000_v2.ply", V1, F1);
    igl::readPLY("../bunny_v2/bun045_v2.ply", V2, F2);


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

    //Take samples from V2 (which change every iteration)
    int numOfSamples = 1000;
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
    cout << "num of samples: " << numOfSamples << endl;
    cout << "num of rows: " << samples.rows() << endl;
    Eigen::MatrixXd samplesNN(samples.rows(), samples.cols());
    for (int sample = 0; sample < numOfSamples; sample++) {
        cout << sample << endl;
        int rowIndex = kNNResults.row(sample)(0);
        samplesNN.row(sample) = V1.row(rowIndex);
    }
    cout << "passed collecting points" << endl;
    samplesG = samples;
    samplesNNG = samplesNN;

    Eigen::MatrixXd R;
    Eigen::RowVector3d t;

    computeRt(R, t, samples, samplesNN);

    //Computing barycenters for samples and its nn
    Eigen::RowVector3d samplesBC = samples.colwise().sum()/samples.rows();
    Eigen::RowVector3d samplesNNBC = samplesNN.colwise().sum()/samplesNN.rows();

    viewer.data().add_points(samplesBC, Eigen::RowVector3d(1, 0, 0));
    viewer.data().add_points(samplesNNBC, Eigen::RowVector3d(0, 1, 0))

    //Compute the recentered points
    Eigen::MatrixXd hatSamples = samples.rowwise() - samplesBC;
    Eigen::MatrixXd hatSamplesNN = samplesNN.rowwise() - samplesNNBC;

    viewer.data().add_points(hatSamples, Eigen::RowVector3d(1,1,1));
    viewer.data().add_points(hatSamplesNN, Eigen::RowVector3d(0,0,0));

    viewer.data().point_size = point_size;
    viewer.data().add_points(V1, Eigen::RowVector3d(1, 0, 0.5)); //pink
    viewer.data().add_points(currV2, Eigen::RowVector3d(0.4, 0.5, 0.1)); //greenish
    viewer.callback_key_down =  &key_down;
    viewer.launch();

    return 0;
}
