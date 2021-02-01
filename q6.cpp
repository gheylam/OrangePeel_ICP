#include <stdlib.h>
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

int nNumBuns = 1;
int point_size = 2;
int displaying = 0;
std::vector<Eigen::MatrixXd> iterations;
std::vector<Eigen::MatrixXd> vBunPts;
std::vector<Eigen::MatrixXi> vBunFaces;
std::vector<Eigen::RowVector3d> vColours;

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
            displaying-=nNumBuns;
            cout << "displaying iteration: " << displaying << endl;
            viewer.data().clear();
            viewer.data().add_points(iterations[displaying], vColours[1]);
            viewer.data().add_points(vBunPts[0], vColours[0]);
        }
    }else if(key == '4'){
        if(displaying < iterations.size()){
            displaying+=nNumBuns;
            cout << "displaying iteration: " << displaying << endl;
            viewer.data().clear();
            viewer.data().add_points(iterations[displaying], vColours[1]);
            viewer.data().add_points(vBunPts[0], vColours[0]);
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
        ptSamples = points;
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

void addZeroMeanNoise(Eigen::MatrixXd& outputM, Eigen::MatrixXd& inputM, double sd){
    if(sd == 0){
        outputM = inputM;
        return;
    }
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, sd);
    Eigen::MatrixXd rndMatrix = Eigen::MatrixXd::Ones(inputM.rows(), inputM.cols());
    for(int rowId = 0; rowId < inputM.rows(); rowId++){
        rndMatrix.row(rowId) = rndMatrix.row(rowId) * (distribution(generator));
    }
    rndMatrix = rndMatrix + inputM;
    outputM = rndMatrix;
}

void computePseudoInverse(Eigen::MatrixXd& mOutputM, Eigen::MatrixXd& A){
    //std::cout << "A" << std::endl;
    //std::cout << A << std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd vSingularValues = svd.singularValues();
    Eigen::MatrixXd mInvertedSigma = Eigen::MatrixXd::Zero(A.cols(), A.rows());

    for(int nValID = 0; nValID < vSingularValues.size(); nValID++){
        double dCoeff = 1 / vSingularValues(nValID);
        mInvertedSigma(nValID, nValID) = dCoeff;
    }
    Eigen::MatrixXd mPInv = Eigen::MatrixXd::Zero(A.cols(), A.rows());
    mPInv = svd.matrixV() * mInvertedSigma * svd.matrixU().transpose();
    //std::cout << "Pseudo Inverse" << std::endl;
    //std::cout << mPInv << std::endl;
    mOutputM = mPInv;
}

/*
 * This program is used for experimenting how sampling affects the
 * accurary of the ICP convergence
 */

int main(int argc, char* args[]){

    /*
     * q6 - point to plane ICP implementation
     * This code's intention is to use the approximation to the minimum least squares
     * solution of the point to plane ICP implementation. I am using the derivation
     * from the literation referece: [1] K. Low, “Linear Least-squares Optimization for
     * Point-to-plane ICP Surface Registration,” Chapel Hill, Univ. North Carolina, no.
     * February, pp. 2–4, 2004, [Online].
     * Available: https://www.iscs.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf.
     *
     * Impementation overview:
     * 1. Define subject bunny and target bunny
     * 2. target bunny will remain static and so we will create the Octree from the target bunny
     * 3. Sample the subject bunny
     * 4. Find corresponding points from samples and get the corresponding point's normals
     * 5. Reject outliers
     * 6. Construct design matrix A
     * 7. Construct matrix b
     * 8. Compute SVD of A
     * 9. Compute Pseudo inverse of A using SVD results
     * 10. Find the optimal parameters by multiplying Pseudo inverse of A with B
     * 11. Construct Rotation matrix R and translation vector t
     * 12. Apply transformation on subject bunny
     * 13. Repeat
     */




    //1.Define the subject bunny and target bunny
    //Load in the non transformed bunny
    std::vector<std::string> filepaths;
    filepaths.push_back("../bunny_v2/bun000_v2.ply");
    filepaths.push_back("../bunny_v2/bun045_v2.ply");
    //filepaths.push_back("../bunny_v2/bun090_v2.ply");
    //filepaths.push_back("../bunny_v2/bun180_v2.ply");
    //filepaths.push_back("../bunny_v2/bun270_v2.ply");
    //filepaths.push_back("../bunny_v2/bun315_v2.ply");

    nNumBuns = 1;

    //Specify colours for each bunny
    vColours.push_back(Eigen::RowVector3d(1, 0, 0));
    vColours.push_back(Eigen::RowVector3d(0, 1, 0));

    for(int i = 0; i < filepaths.size(); i++){
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readPLY(filepaths[i], V, F);
        vBunPts.push_back(V);
        vBunFaces.push_back(F);
    }

    int nTargetBunny = 0;
    int nSubjectBunny = 1;

    iterations.push_back(vBunPts[nSubjectBunny]);

    //Get the normals of the Target Bunny
    Eigen::MatrixXd mTargetNormals;
    igl::per_vertex_normals(vBunPts[nTargetBunny], vBunFaces[nTargetBunny], mTargetNormals);

    //2.Construct Octree of the static TargetBunny
    std::vector<std::vector<int>> O_PI;
    Eigen::MatrixXi O_CH;
    Eigen::MatrixXd O_CN;
    Eigen::MatrixXd O_W;
    igl::octree(vBunPts[nTargetBunny], O_PI, O_CH, O_CN, O_W);

    //Parameters for the iterations
    int nNumSamp = 1000;
    int nMaxIter = 100;
    double dAcceptThreshold = 0.8;
    double dConvergenceThreshold = 0.0001;
    double dLastDiff = 999999999;

    if(argc > 1){
        nMaxIter = std::stoi(args[1]);
        dConvergenceThreshold = std::stod(args[2]);
    }


    //3.Sample SubjectBunny and Get each sample's normal as well
    for(int iter = 0; iter < nMaxIter; iter++) {
        Eigen::MatrixXd mSamples = Eigen::MatrixXd::Zero(nNumSamp, 3);
        getSamples(mSamples, vBunPts[nSubjectBunny], nNumSamp);

        //4.Find the corresponding points
        Eigen::MatrixXi mKNNResults;
        igl::knn(mSamples, vBunPts[nTargetBunny], 1, O_PI, O_CH, O_CN, O_W, mKNNResults);

        //Construct mSampleNN (The corresponding samples from target bunny)
        //Also construct mSampleNormals (The normals of the corresponding samples from target bunny)
        Eigen::MatrixXd mSamplesNN(nNumSamp, 3);
        Eigen::MatrixXd mSampleNormals(nNumSamp, 3);
        for (int s = 0; s < nNumSamp; s++) {
            int nRowIndex = mKNNResults.row(s)(0);
            mSamplesNN.row(s) = vBunPts[nTargetBunny].row(nRowIndex);
            mSampleNormals.row(s) = mTargetNormals.row(nRowIndex);
        }

        //5.Reject outliers that are too far apart
        Eigen::MatrixXd mSampleDists = (mSamples - mSamplesNN).cwiseAbs2().rowwise().sum();
        std::vector<int> vKeepers;
        double dAvgDist = mSampleDists.sum() / nNumSamp;
        for (int s = 0; s < nNumSamp; s++) {
            if (mSampleDists(s) < (dAvgDist * dAcceptThreshold)) {
                if (!isnan(mSampleNormals.row(s)(0))) {
                    vKeepers.push_back(s);
                }
            }
        }
        Eigen::MatrixXd mKeptSamples = Eigen::MatrixXd::Zero(vKeepers.size(), 3);
        Eigen::MatrixXd mKeptSamplesNN = Eigen::MatrixXd::Zero(vKeepers.size(), 3);
        Eigen::MatrixXd mKeptSampleNormals = Eigen::MatrixXd::Zero(vKeepers.size(), 3);

        for (int nKeeperId = 0; nKeeperId < vKeepers.size(); nKeeperId++) {
            int rowIndex = vKeepers[nKeeperId];
            mKeptSamples.row(nKeeperId) = mSamples.row(rowIndex);
            mKeptSamplesNN.row(nKeeperId) = mSamplesNN.row(rowIndex);
            mKeptSampleNormals.row(nKeeperId) = mSampleNormals.row(rowIndex);
        }

        //6. Build the Design Matrix A

        Eigen::MatrixXd mA = Eigen::MatrixXd::Zero(vKeepers.size(), 6);
        for (int rowId = 0; rowId < vKeepers.size(); rowId++) {
            Eigen::RowVector3d rvSi = mKeptSamples.row(rowId);
            Eigen::RowVector3d rvNi = mKeptSampleNormals.row(rowId);
            int x = 0;
            int y = 1;
            int z = 2;
            double dCol1 = rvNi(z) * rvSi(y) - rvNi(y) * rvSi(z); // NizSiy - NiySiz
            double dCol2 = rvNi(x) * rvSi(z) - rvNi(z) * rvSi(x); // NixSiz - NizSix
            double dCol3 = rvNi(y) * rvSi(x) - rvNi(x) * rvSi(y); // NiySix - NixSiy
            double dCol4 = rvNi(x);
            double dCol5 = rvNi(y);
            double dCol6 = rvNi(z);
            Eigen::RowVectorXd rvA(6);
            rvA << dCol1, dCol2, dCol3, dCol4, dCol5, dCol6;
            mA.row(rowId) = rvA;
        }

        //7. Build vector B - Keep in mind that this is not a RowVector (its a column vector)
        Eigen::VectorXd vB = Eigen::VectorXd::Zero(vKeepers.size());
        for (int rowId = 0; rowId < vKeepers.size(); rowId++) {
            Eigen::RowVector3d rvSi = mKeptSamples.row(rowId);
            Eigen::RowVector3d rvDi = mKeptSamplesNN.row(rowId);
            Eigen::RowVector3d rvNi = mKeptSampleNormals.row(rowId);
            int x = 0;
            int y = 1;
            int z = 2;
            double dRow =
                    rvNi(x) * rvDi(x) + rvNi(y) * rvDi(y) + rvNi(z) * rvDi(z) - rvNi(x) * rvSi(x) - rvNi(y) * rvSi(y) -
                    rvNi(z) * rvSi(z);
            vB(rowId) = dRow;
        }

        //8 & 9 Compute SVD and then construct the pseudo inverse of A
        Eigen::MatrixXd mPInv;
        computePseudoInverse(mPInv, mA);

        //10. Find optimal parameters
        Eigen::VectorXd vParams = Eigen::VectorXd::Zero(6);
        vParams = mPInv * vB;
        double x = vParams(3);
        double y = vParams(4);
        double z = vParams(5);
        Eigen::RowVector3d t(x, y, z);
        Eigen::Matrix3d R;
        double alpha = vParams(0);
        double beta = vParams(1);
        double gamma = vParams(2);
        R << 1, alpha*beta-gamma, alpha*gamma+beta,
             gamma, alpha*beta*gamma+1, beta*gamma-alpha,
             -beta, alpha, 1;

        vBunPts[nSubjectBunny] = (R * vBunPts[nSubjectBunny].transpose()).transpose().rowwise() + t;

        iterations.push_back(vBunPts[nSubjectBunny]);
        cout << "Completed Iteration: " << iter << endl;
        double dThisDiff = computeSqrDiff(mKeptSamples, mKeptSamplesNN);
        if(abs(dThisDiff - dLastDiff) < dConvergenceThreshold){
            cout << "Converged with error: " << dThisDiff << endl;
            cout << "Last diff: " << dLastDiff << endl;
            cout << "Iteration diff: " << abs(dThisDiff-dLastDiff) << endl;
            break;
        }
        dLastDiff = dThisDiff;
    }
    //Create viewer for displaying the file
    igl::opengl::glfw::Viewer viewer;

    //FUN: Playing with Vertex normal generated edges
    /*
     *  //Get the vertex normals
        Eigen::MatrixXd V1_N;
        igl::per_vertex_normals(vBunPts[0], vBunFaces[0], V1_N);

        Eigen::MatrixXd V1_N_edges = Eigen::MatrixXd::Zero(vBunPts[0].rows(), 3);
        double normal_length = 0.001;
        if(argc > 1){
            normal_length = std::stod(args[1]);
        }

        V1_N_edges = vBunPts[0] + (V1_N * normal_length);


        viewer.data().add_edges(vBunPts[0], V1_N_edges, V1_N);
    */

    viewer.data().add_points(vBunPts[nTargetBunny], vColours[0]);
    viewer.data().add_points(vBunPts[nSubjectBunny], vColours[1]);

    viewer.data().point_size = point_size;
    viewer.callback_key_down =  &key_down;
    viewer.launch();

    return 0;
}
