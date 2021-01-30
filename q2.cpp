#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <igl/readPLY.h>
#include <igl/knn.h>
#include <igl/octree.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/get_seconds.h>
#include "gnuplot_i.h"
#include <math.h>

/*
 * This .cpp file is for experimenting the behaviour of the ICP
 * algorithm's convergence rate based on the extremity of the
 * rotation of M1 points
 */

using namespace std;
Eigen::MatrixXd V1;
Eigen::MatrixXi F1;
int point_size = 10;
int displaying = 0;
int frame = 0;
vector<Eigen::MatrixXd> iterations;
vector<vector<Eigen::MatrixXd>> iterTransformations;
vector<double> sqrDiff;
vector<int>convergenceRate;
vector<int>degreesSeq;


void makeRigidHomo(Eigen::MatrixXd& rigidTransform, Eigen::MatrixXd& R, Eigen::RowVector3d& t){
    /*
     *Assembles the Rigid transformation using homogenous coordinates
     */
    Eigen::MatrixXd rigidHomo = Eigen::MatrixXd ::Zero(4, 4);
    rigidHomo.block(0, 0, 3, 3) = R;
    rigidHomo.block(0, 3, 3, 1) = t.transpose();
    rigidHomo(3, 3) = 1;
    rigidTransform = rigidHomo;
}

void makeHomo(Eigen::MatrixXd& homoPts, Eigen::MatrixXd& pts){
    Eigen::MatrixXd homo = Eigen::MatrixXd::Zero(pts.rows()+1, pts.cols());
    homo.block(0, 0, pts.rows(), pts.cols()) = pts;
    homo.block(homo.rows()-1, 0, 1, homo.cols()) = Eigen::RowVectorXd::Ones(homo.cols());
    homoPts = homo;
}

void makeCart(Eigen::MatrixXd& cartPts, Eigen::MatrixXd& homoPts){
    Eigen::MatrixXd cart = Eigen::MatrixXd::Zero(homoPts.rows()-1, homoPts.cols());
    cart = homoPts.block(0, 0, cart.rows(), cart.cols());
    cartPts = cart;
}

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
            for(int p = 0; p < 20; p++){
                viewer.data().add_points(Eigen::RowVector3d(0,0,p), Eigen::RowVector3d(0, 1, 0));
                viewer.data().add_points(Eigen::RowVector3d(0,p,0), Eigen::RowVector3d(0, 0, 1));
                viewer.data().add_points(Eigen::RowVector3d(p,0,0), Eigen::RowVector3d(1, 0, 0));
            }
        }
    }else if(key == '4'){
        if(displaying != (iterations.size()-1)){
            displaying++;
            cout << "displaying iteration: " << displaying << endl;
            viewer.data().clear();
            viewer.data().add_points(V1, Eigen::RowVector3d(1, 0, 0.5));
            viewer.data().add_points(iterations[displaying], Eigen::RowVector3d(0.4, 0.5, 0.1));
            for(int p = 0; p < 20; p++){
                viewer.data().add_points(Eigen::RowVector3d(0,0,p), Eigen::RowVector3d(0, 1, 0));
                viewer.data().add_points(Eigen::RowVector3d(0,p,0), Eigen::RowVector3d(0, 0, 1));
                viewer.data().add_points(Eigen::RowVector3d(p,0,0), Eigen::RowVector3d(1, 0, 0));
            }
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

void computeRotationZ(Eigen::MatrixXd& rotationM, double degrees){
    double radians = (degrees / 180) * M_PI;
    Eigen::MatrixXd rotationZ = Eigen::MatrixXd::Zero(3, 3);
    rotationZ << cos(radians), -sin(radians), 0,
            sin(radians), cos(radians), 0,
            0, 0, 1;

    cout << rotationZ << endl;
    rotationM = rotationZ;
}

void computeRotationX(Eigen::MatrixXd& rotationM, double degrees){
    double radians = (degrees / 180) * M_PI;
    Eigen::MatrixXd rotationX = Eigen::MatrixXd::Zero(3, 3);
    rotationX << 1, 0, 0,
                 0, cos(radians), -sin(radians),
                 0, sin(radians), cos(radians);

    cout << rotationX << endl;
    rotationM = rotationX;
}

void computeRotationY(Eigen::MatrixXd& rotationM, double degrees){
    double radians = (degrees / 180) * M_PI;
    Eigen::MatrixXd rotationY = Eigen::MatrixXd::Zero(3, 3);
    rotationY << cos(radians), 0, sin(radians),
            0, 1, 0,
            -sin(radians), 0, cos(radians);

    cout << rotationY << endl;
    rotationM = rotationY;
}



int main(int argc, char* args[]){
    //Load in the non transformed bunny
    igl::readPLY("../bunny_v2/bun000_v2.ply", V1, F1);
    //center the bunny at (0,0,0)
    /*
     * 1. Compute centroid of the bunny
     * 2. Subtract all points of the bunn by the centroid
     */
    Eigen::RowVector3d v1Centroid = V1.colwise().sum() / V1.rows();
    Eigen::MatrixXd centeredV1 = V1.rowwise() - v1Centroid;
    V1 = centeredV1;

    vector<int> iterationsDone;
    int numDTrials = 10;
    int icpIter = 100;
    int increment = 1;
    int numOfSamples = 1000;
    double acceptanceThreshold = 5;

    if(argc > 1){
        numDTrials = stoi(args[1]);
        increment = stoi(args[2]);
    }

    //Construct our octree out of our Q Points (that we are trying to map to)
    //Initialize Octree properties
    std::vector<std::vector<int>> O_PI; //Points inside each Octree cell
    Eigen::MatrixXi O_CH; //Children of each node in the Octree
    Eigen::MatrixXd O_CN; //Mid point of each node in the Octree
    Eigen::VectorXd O_W; //Width of each node in the Octree
    //Compute Octree
    igl::octree(V1, O_PI, O_CH, O_CN, O_W);

    for(int d = 1; d < numDTrials*increment; d+=increment){
        //construct rotation matrix
        Eigen::MatrixXd rotationM;
        computeRotationY(rotationM, d);
        //Rotate V1 by d degrees
        Eigen::MatrixXd V2 = (rotationM * V1.transpose()).transpose();

        for(int i = 0; i < icpIter; i++) {
            //Take samples from V2 (which change every iteration)
            Eigen::MatrixXd samples(numOfSamples, 3);
            getSamples(samples, V2, numOfSamples);

            //We have our set of points P
            //Next we need to find the closest point for each p in Q

            //Set up results matrix for kNN
            Eigen::MatrixXi kNNResults; //This will hold the kNN for each point we give to the algorithm
            double tic = igl::get_seconds();
            igl::knn(samples, V1, 1, O_PI, O_CH, O_CN, O_W, kNNResults);
            printf("%0.4f secs\n",igl::get_seconds()-tic);

            //Collect the V1 points found
            Eigen::MatrixXd samplesNN(samples.rows(), samples.cols());
            for (int sample = 0; sample < numOfSamples; sample++) {
                int rowIndex = kNNResults.row(sample)(0);
                samplesNN.row(sample) = V1.row(rowIndex);
            }
            cout << "passed collecting points" << endl;

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
            V2 = (R*V2.transpose()).transpose().rowwise() + t;
            cout << "Iteration number: " << i << endl;
            iterations.push_back(V2);
            double epsilon = computeSqrDiff(V2, V1);
            sqrDiff.push_back(epsilon);
            if(i != 0 && abs(sqrDiff[i-1] - epsilon) < 0.001){
                iterationsDone.push_back(i);
                break;
            }
        }

    }
    Gnuplot g1("lines");
    g1.plot_x(iterationsDone, "ConvergenceRate");
    g1.showonscreen();
    wait_for_key();

    //Create viewer for displaying the file
    igl::opengl::glfw::Viewer viewer;

    viewer.data().point_size = point_size;
    viewer.data().add_points(V1, Eigen::RowVector3d(1, 0, 0.5)); //pink
    viewer.data().add_points(iterations[0], Eigen::RowVector3d(0.4, 0.5, 0.1)); //greenish
    viewer.callback_key_down =  &key_down;
    viewer.launch();

    return 0;
}
