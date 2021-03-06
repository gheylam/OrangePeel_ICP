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
#include <random>

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

//Function prototypes
void addMeanZeroNoise(Eigen::MatrixXd& outputM, Eigen::MatrixXd& inputM, double sd);


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
    }else if(key == '4') {
        if (displaying != (iterations.size() - 1)) {
            displaying++;
            cout << "displaying iteration: " << displaying << endl;
            viewer.data().clear();
            viewer.data().add_points(V1, Eigen::RowVector3d(1, 0, 0.5));
            viewer.data().add_points(iterations[displaying], Eigen::RowVector3d(0.4, 0.5, 0.1));
            for (int p = 0; p < 20; p++) {
                viewer.data().add_points(Eigen::RowVector3d(0, 0, p), Eigen::RowVector3d(0, 1, 0));
                viewer.data().add_points(Eigen::RowVector3d(0, p, 0), Eigen::RowVector3d(0, 0, 1));
                viewer.data().add_points(Eigen::RowVector3d(p, 0, 0), Eigen::RowVector3d(1, 0, 0));
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

double computeAbsDiff(Eigen::MatrixXd& samples, Eigen::MatrixXd& samplesNN){
    double sqrDiff = (samples - samplesNN).cwiseAbs().sum();
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

void addZeroMeanNoise(Eigen::MatrixXd& outputM, Eigen::MatrixXd& inputM, double sd){
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, sd);
    Eigen::MatrixXd rndMatrix = Eigen::MatrixXd::Ones(inputM.rows(), inputM.cols());
    for(int rowId = 0; rowId < inputM.rows(); rowId++){
        rndMatrix.row(rowId) = rndMatrix.row(rowId) * (distribution(generator));
    }
    rndMatrix = rndMatrix + inputM;
    outputM = rndMatrix;
}

void computeBoundaryBox(Eigen::MatrixXd& boundaryPoints, Eigen::MatrixXd& mesh){
    //Finds the boundary box coordinates of a mesh
    Eigen::MatrixXd pts = Eigen::MatrixXd::Zero(8, 3);
    //Min and Max of X-axis
    double minX = mesh.col(0).minCoeff();
    double maxX = mesh.col(0).maxCoeff();
    //Min and Max of Y-axis
    double minY = mesh.col(1).minCoeff();
    double maxY = mesh.col(1).maxCoeff();
    //Min and Max of Z-axis
    double minZ = mesh.col(2).minCoeff();
    double maxZ = mesh.col(2).maxCoeff();

    pts.row(0) = Eigen::RowVector3d(minX, minY, minZ);
    pts.row(1) = Eigen::RowVector3d(minX, maxY, minZ);
    pts.row(2) = Eigen::RowVector3d(maxX, maxY, minZ);
    pts.row(3) = Eigen::RowVector3d(maxX, minY, minZ);
    pts.row(4) = Eigen::RowVector3d(minX, minY, maxZ);
    pts.row(5) = Eigen::RowVector3d(minX, maxY, maxZ);
    pts.row(6) = Eigen::RowVector3d(maxX, maxY, maxZ);
    pts.row(7) = Eigen::RowVector3d(maxX, minY, maxZ);

    boundaryPoints = pts;
}

void computeBoundaryEdges(Eigen::MatrixXd& edgePts1, Eigen::MatrixXd& edgePt2, Eigen::MatrixXd& boundaryPts){
    Eigen::MatrixXd pts = Eigen::MatrixXd::Zero(12, 3);
    Eigen::MatrixXd matchingPts = Eigen::MatrixXd::Zero(12, 3);
    pts.block(0, 0, 8, 3) = boundaryPts;
    pts.block(8, 0, 4, 3) = boundaryPts.block(0, 0, 4, 3);

    matchingPts.row(0) = boundaryPts.row(1);
    matchingPts.row(1) = boundaryPts.row(2);
    matchingPts.row(2) = boundaryPts.row(3);
    matchingPts.row(3) = boundaryPts.row(0);
    matchingPts.row(4) = boundaryPts.row(5);
    matchingPts.row(5) = boundaryPts.row(6);
    matchingPts.row(6) = boundaryPts.row(7);
    matchingPts.row(7) = boundaryPts.row(4);
    matchingPts.row(8) = boundaryPts.row(4);
    matchingPts.row(9) = boundaryPts.row(5);
    matchingPts.row(10) = boundaryPts.row(6);
    matchingPts.row(11) = boundaryPts.row(7);

    edgePts1 = pts;
    edgePt2 = matchingPts;
}

int main(int argc, char* args[]){
    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;
    Eigen::MatrixXd V3;
    Eigen::MatrixXi F3;
    Eigen::MatrixXd V4;
    Eigen::MatrixXi F4;
    Eigen::MatrixXd V5;
    Eigen::MatrixXi F5;
    Eigen::MatrixXd V6;
    Eigen::MatrixXi F6;
    //Load in the non transformed bunny
    igl::readPLY("../bunny_v2/bun000_v2.ply", V1, F1);
    igl::readPLY("../bunny_v2/bun045_v2.ply", V2, F2);
    igl::readPLY("../bunny_v2/bun090_v2.ply", V3, F3);
    igl::readPLY("../bunny_v2/bun180_v2.ply", V4, F4);
    igl::readPLY("../bunny_v2/bun270_v2.ply", V5, F5);
    igl::readPLY("../bunny_v2/bun315_v2.ply", V6, F6);

    //Rotate extreme bunnies closer to center bunny


    //Create viewer for displaying the file
    igl::opengl::glfw::Viewer viewer;

    //Testing rotation method
    /*
     * Displaying axis and centering bunny
    Eigen::RowVector3d v1Centroid = V1.colwise().sum() / V1.rows();
    Eigen::MatrixXd centeredV1 = V1.rowwise() - v1Centroid;
    V1 = centeredV1;
    Eigen::MatrixXd rotation;
    Eigen::MatrixXd currV2 = centeredV1;
    //Z makes it spin palm facing towards me
    for(int degrees = -90; degrees < 91; degrees++) {
        computeRotationZ(rotation, degrees);
        currV2 = (rotation * currV2.transpose()).transpose();
        iterations.push_back(currV2);
    }
    //X makes its spin fingers facing towards me
    currV2 = V1;
    for(int degrees = -90; degrees < 91; degrees++) {
        computeRotationX(rotation, degrees);
        currV2 = (rotation * currV2.transpose()).transpose();
        iterations.push_back(currV2);
    }
    //Y makes its spin thumb towards me
    currV2 = V1;
    for(int degrees = -90; degrees < 91; degrees++) {
        computeRotationY(rotation, degrees);
        currV2 = (rotation * currV2.transpose()).transpose();
        iterations.push_back(currV2);
    }




    for(int p = 0; p < 20; p++){
        viewer.data().add_points(Eigen::RowVector3d(0,0,p), Eigen::RowVector3d(0, 1, 0));
        viewer.data().add_points(Eigen::RowVector3d(0,p,0), Eigen::RowVector3d(0, 0, 1));
        viewer.data().add_points(Eigen::RowVector3d(p,0,0), Eigen::RowVector3d(1, 0, 0));
    }

    viewer.data().add_points(Eigen::RowVector3d(0, 0, 0), Eigen::RowVector3d(1, 0, 0));
    viewer.data().point_size = point_size;
    viewer.data().add_points(centeredV1, Eigen::RowVector3d(1, 0, 0.5)); //pink
    viewer.data().add_points(iterations[0], Eigen::RowVector3d(0.4, 0.5, 0.1)); //greenish
    */
    //Testing adding noise method
    /*
     * The following is for testing the noise function
        double sd = 0.1;
        if(argc > 1){
            sd = std::stod(args[1]);

        }
        //Adding noise to V1 and storing the noised points into V2
        addZeroMeanNoise(V2, V1, sd);
    */
    //Testing boundary boxes
    //Find the boundary box of a mesh
    Eigen::MatrixXd mBoundaryV1;
    Eigen::MatrixXd mBoundaryV2;
    Eigen::MatrixXd mBoundaryV3;
    Eigen::MatrixXd mBoundaryV4;
    Eigen::MatrixXd mBoundaryV5;
    Eigen::MatrixXd mBoundaryV6;
    computeBoundaryBox(mBoundaryV1, V1);
    computeBoundaryBox(mBoundaryV2, V2);
    computeBoundaryBox(mBoundaryV3, V3);
    computeBoundaryBox(mBoundaryV4, V4);
    computeBoundaryBox(mBoundaryV5, V5);
    computeBoundaryBox(mBoundaryV6, V6);

    Eigen::MatrixXd edge1_V1;
    Eigen::MatrixXd edge2_V1;
    Eigen::MatrixXd edge1_V2;
    Eigen::MatrixXd edge2_V2;
    Eigen::MatrixXd edge1_V3;
    Eigen::MatrixXd edge2_V3;
    Eigen::MatrixXd edge1_V4;
    Eigen::MatrixXd edge2_V4;

    computeBoundaryEdges(edge1_V1, edge2_V1, mBoundaryV1);
    computeBoundaryEdges(edge1_V2, edge2_V2, mBoundaryV2);
    computeBoundaryEdges(edge1_V3, edge2_V3, mBoundaryV3);
    computeBoundaryEdges(edge1_V4, edge2_V4, mBoundaryV4);

    //plot the boundary box
    viewer.data().add_points(mBoundaryV1, Eigen::RowVector3d(1,0,0));
    viewer.data().add_points(V1, Eigen::RowVector3d(0, 0.8, 0.2));
    viewer.data().add_edges(edge1_V1, edge2_V1, Eigen::RowVector3d(1, 0, 0));

    viewer.data().add_points(mBoundaryV2, Eigen::RowVector3d(0,0,1));
    viewer.data().add_points(V2, Eigen::RowVector3d(1, 0.8, 0.2));
    viewer.data().add_edges(edge1_V2, edge2_V2, Eigen::RowVector3d(1, 0, 0));

    viewer.data().add_points(mBoundaryV3, Eigen::RowVector3d(0,1,1));
    viewer.data().add_points(V3, Eigen::RowVector3d(1, 0.5, 1));
    viewer.data().add_edges(edge1_V3, edge2_V3, Eigen::RowVector3d(0, 1, 1));

    viewer.data().add_points(mBoundaryV4, Eigen::RowVector3d(0, 1, 1));
    viewer.data().add_points(V4, Eigen::RowVector3d(0.3, 0.3, 1));
    viewer.data().add_edges(edge1_V4, edge2_V4, Eigen::RowVector3d(0, 1, 1));
    viewer.callback_key_down =  &key_down;
    viewer.launch();

    return 0;
}
