#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <igl/readPLY.h>
#include <igl/knn.h>
#include <igl/octree.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

using namespace std;
Eigen::MatrixXd V1;
Eigen::MatrixXi F1;
int point_size = 10;
int displaying = 0;
vector<Eigen::MatrixXd> iterations;

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

void computeRt(Eigen::MatrixXd& R, Eigen::RowVector3d t, Eigen::MatrixXd& P, Eigen::MatrixXd& Q){
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
    Eigen::RowVector3d samplesBC = P.colwise().sum();
    Eigen::RowVector3d samplesNNBC = Q.colwise().sum();

    //Compute the recentered points
    Eigen::MatrixXd hatSamples = Eigen::MatrixXd::Zero(P.rows(), P.cols());
    Eigen::MatrixXd hatSampleNN = Eigen::MatrixXd::Zero(Q.rows(), Q.cols());

    for(int rowId = 0; rowId < P.rows(); rowId++){
        hatSamples.row(rowId) = P.row(rowId) - samplesBC;
        hatSampleNN.row(rowId) = Q.row(rowId) - samplesNNBC;
    }

    //Assemble matrix A
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
    for(int rowId = 0; rowId < P.rows(); rowId++){
        A = A + (hatSampleNN.row(rowId).transpose() * hatSamples.row(rowId));
    }

    //Compute the SVD of A then assemble R and t
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixV() * svd.matrixU().transpose();
    t = samplesBC - (R * samplesNNBC.transpose()).transpose();
}

int main(int argc, char* args[]){
    //Load in the files

    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;
    igl::readPLY("../bunny_v2/bun000_v2.ply", V1, F1);
    igl::readPLY("../bunny_v2/bun045_v2.ply", V2, F2);


    //Create viewer for displaying the file
    igl::opengl::glfw::Viewer viewer;

    Eigen::MatrixXd currV2 = V2;
    iterations.push_back(V2);
    int iter = 20;
    for(int i = 0; i < iter; i++) {
        //First sample points
        int numOfSamples = V1.rows();
        numOfSamples = 2000;
        Eigen::MatrixXd samples(numOfSamples, 3);
        getSamples(samples, V1, numOfSamples);

        //We have our set of points P
        //Next we need to find the closest point for each p in Q

        //First construct our octree out of our Q Points
        //Initialize Octree properties
        std::vector<std::vector<int>> O_PI; //Points inside each Octree cell
        Eigen::MatrixXi O_CH; //Children of each node in the Octree
        Eigen::MatrixXd O_CN; //Mid point of each node in the Octree
        Eigen::VectorXd O_W; //Width of each node in the Octree

        //Compute Octree
        igl::octree(currV2, O_PI, O_CH, O_CN, O_W);

        //Set up results matrix for kNN
        Eigen::MatrixXi kNNResults; //This will hold the kNN for each point we give to the algorithm
        int numNN = 1;
        igl::knn(samples, currV2, numNN, O_PI, O_CH, O_CN, O_W, kNNResults);
        std::cout << "Correspondences found" << std::endl;
        //Collect the V2 points found
        Eigen::MatrixXd samplesNN(samples.rows(), samples.cols());
        for (int sample = 0; sample < numOfSamples; sample++) {
            int rowIndex = kNNResults.row(sample)(0);
            samplesNN.row(sample) = V2.row(rowIndex);
        }

        //The commented code below is for visualizing the results of the knn
        /*
         * render the sample points and their nn
        viewer.data().add_points(samples, Eigen::RowVector3d(1,1,1));
        //Eigen::MatrixXd nnMinDistance(samples.rows(), numNN);
        for(int k = 0; k < numOfSamples; k++){
            Eigen::RowVectorXi nn = kNNResults.row(k);
            Eigen::RowVector3d rndColour(rand(), rand(), rand());
            rndColour = rndColour / rndColour.sum();
            for(int neighbour = 0; neighbour < nn.size(); neighbour++){
                int index = nn(neighbour);
                viewer.data().add_points(V2.row(index), rndColour);
                double distance = (samples.row(k) - V2.row(index)).array().square().sum();
                nnMinDistance(k, neighbour) = distance;
                V2.row(index) = Eigen::RowVector3d(0,0,0);
            }
        }
        */

        //TODO::Create a outlier rejection phase

        //Alignment phase
        Eigen::MatrixXd R;
        Eigen::RowVector3d t;
        computeRt(R, t, samples, samplesNN);
        currV2 = currV2 * R;
        iterations.push_back(currV2);
        cout << "Iteration number: " << i << endl;
    }
    viewer.data().point_size = point_size;
    viewer.data().add_points(V1, Eigen::RowVector3d(1, 0, 0.5)); //pink
    viewer.data().add_points(currV2, Eigen::RowVector3d(0.4, 0.5, 0.1)); //greenish
    viewer.callback_key_down =  &key_down;
    viewer.launch();

    return 0;
}
