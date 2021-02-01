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

int nNumBuns = 2;
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

            for(int nBunId = 0; nBunId < nNumBuns; nBunId++){
                viewer.data().add_points(iterations[displaying+nBunId], vColours[nBunId]);
            }

        }
    }else if(key == '4'){
        if(displaying != (iterations.size()/nNumBuns)){
            displaying+=nNumBuns;
            cout << "displaying iteration: " << displaying << endl;
            viewer.data().clear();
            for(int nBunId = 0; nBunId < nNumBuns; nBunId++){
                viewer.data().add_points(iterations[displaying+nBunId], vColours[nBunId]);
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

/*
 * This program is used for experimenting how sampling affects the
 * accurary of the ICP convergence
 */


int main(int argc, char* args[]){
    //Load in the non transformed bunny
    std::vector<std::string> filepaths;
    filepaths.push_back("../bunny_v2/bun000_v2.ply");
    filepaths.push_back("../bunny_v2/bun045_v2.ply");
    //filepaths.push_back("../bunny_v2/bun090_v2.ply");
    //filepaths.push_back("../bunny_v2/bun180_v2.ply");
    filepaths.push_back("../bunny_v2/bun270_v2.ply");
    filepaths.push_back("../bunny_v2/bun315_v2.ply");

    nNumBuns = 4;

    //Specify colours for each bunny
    vColours.push_back(Eigen::RowVector3d(1, 0, 0));
    vColours.push_back(Eigen::RowVector3d(0, 1, 0));
    vColours.push_back(Eigen::RowVector3d(0, 0, 1));
    vColours.push_back(Eigen::RowVector3d(1, 1, 0));
    vColours.push_back(Eigen::RowVector3d(1, 0, 1));
    vColours.push_back(Eigen::RowVector3d(0, 1, 1));

    for(int i = 0; i < filepaths.size(); i++){
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readPLY(filepaths[i], V, F);
        vBunPts.push_back(V);
        vBunFaces.push_back(F);
    }

    //Making adjustments to the 270 mesh
    Eigen::MatrixXd R;
    computeRotationY(R, -50);
    vBunPts[2] = (R*vBunPts[2].transpose()).transpose();


    //Create viewer for displaying the file
    igl::opengl::glfw::Viewer viewer;

    /*Display all the bunnies together

    for(auto &pts : vBunPts){
        viewer.data().add_points(pts, Eigen::RowVector3d(1, 0, 1));
    }
    /*

    /*
     * Global registration algorithm
     * 1. Pick one bunny to be subject bunny
     * 2. Build new Octree out of picked bunny
     * 3. Build point cloud out of other bunnies, we denote BigBunny
     * 4. Sample from BigBunny
     * 5. Find corresponding points in subject bunny
     * 6. Compute R and t and shift the subject bunny
     * 7.Iterate 1 - 7
     */

    //Parameters for q5
    double dConvergeThreshold = 0.001;
    int iters = 1;
    if(argc > 1){
        iters = std::stoi(args[1]);
    }
    int nMaxBunSelection = vBunPts.size() * iters;
    double dAcceptThreshold = 0.8;
    int aSqrDiff[nNumBuns];

    //Loop for picking the subject bunny
    for(int bunNum = 0; bunNum < nMaxBunSelection; bunNum++){
        int bunPicked = bunNum % vBunPts.size();
        //int bunPicked = 0;
        cout << "Bun picked: " << bunPicked << endl;
        double tic = igl::get_seconds();

        //Build new octree out of the picked bunny
        std::vector<std::vector<int>> O_PI;
        Eigen::MatrixXi O_CH;
        Eigen::MatrixXd O_CN;
        Eigen::MatrixXd O_W;
        igl::octree(vBunPts[bunPicked], O_PI, O_CH, O_CN, O_W);

        //Build BigBunny
        //First we need to find out the size of bunBunny
        int nBigBunnyRows = 0;
        for(int bunId = 0; bunId < vBunPts.size(); bunId++){
            if(bunId != bunPicked){
                nBigBunnyRows += vBunPts[bunId].rows();
            }
        }
        Eigen::MatrixXd mBigBunnyPts(nBigBunnyRows, 3);
        int nBigBunnyNextRow = 0;
        for(int bunId = 0; bunId < vBunPts.size(); bunId++){
            if(bunId != bunPicked){
                mBigBunnyPts.block(nBigBunnyNextRow, 0, vBunPts[bunId].rows(), 3) = vBunPts[bunId];
                nBigBunnyNextRow += vBunPts[bunId].rows();

            }
        }

        //Sample from mBigBunny pts
        int nNumSamples = mBigBunnyPts.rows()*0.1;
        cout << "Sample size: " << nNumSamples << endl;
        Eigen::MatrixXd samples(nNumSamples, 3);
        getSamples(samples, mBigBunnyPts, nNumSamples);

        //Find corresponding points
        Eigen::MatrixXi mKNNResults;
        igl::knn(samples, vBunPts[bunPicked], 1, O_PI, O_CH, O_CN, O_W, mKNNResults);

        //Produce sampleNN
        Eigen::MatrixXd samplesNN(nNumSamples, 3);
        for(int s = 0; s < nNumSamples; s++){
            int nRowIndex = mKNNResults.row(s)(0);
            samplesNN.row(s) = vBunPts[bunPicked].row(nRowIndex);
        }

        //Reject outliers that are too far apart
        Eigen::MatrixXd mSampleDists = (samples - samplesNN).cwiseAbs2().rowwise().sum();
        std::vector<int> vKeepers;
        double dAvgDist = mSampleDists.sum() / nNumSamples;
        for(int s = 0; s < nNumSamples; s++){
            if(mSampleDists(s) < (dAvgDist * dAcceptThreshold)){
                vKeepers.push_back(s);
            }
        }
        Eigen::MatrixXd mKeptSamples = Eigen::MatrixXd::Zero(vKeepers.size(), 3);
        Eigen::MatrixXd mKeptSamplesNN = Eigen::MatrixXd::Zero(vKeepers.size(), 3);

        for(int nKeeperId = 0; nKeeperId < vKeepers.size(); nKeeperId++){
            int rowIndex = vKeepers[nKeeperId];
            mKeptSamples.row(nKeeperId) = samples.row(rowIndex);
            mKeptSamplesNN.row(nKeeperId) = samplesNN.row(rowIndex);
        }

        //Align the two point clouds
        Eigen::MatrixXd R;
        Eigen::RowVector3d t;
        //ComputeRt is a R, t, fromPts, toPts
        //In the previous programs we would use KeptSamples to KeptSamplesNN because we were
        //after to keep a single mesh fixed. But in global registration, we need to adjust every
        //mesh.
        computeRt(R, t, mKeptSamplesNN, mKeptSamples);
        vBunPts[bunPicked] = (R*vBunPts[bunPicked].transpose()).transpose().rowwise() + t;



        cout << "adjusted bunnyNo: " << bunPicked << endl;
        printf("%0.4f secs\n",igl::get_seconds()-tic);
        for(auto &pts : vBunPts) {
            iterations.push_back(pts);
        }

        /*
         * We need some kind of exit condition. Previous we would use the sqr difference between
         * to different meshes. Now that we have multiple meshes it could be possible that some meshes
         * are adjusting greater amounts compared to other. So my exit condition is going to measure the
         * difference between a subject bunny compared to the bigbunny. This means we can only terminate
         * after adjusting every bunny. I.e. the number of iterations will be constrained to a multiple of
         * nNumBuns
         */
        //Transform the the sampled points and see how well they have converged.
        Eigen::MatrixXd mKeptSamplesNNTransformed = (R*mKeptSamplesNN.transpose()).transpose().rowwise() + t;
        double dSmolBunToBigBun = computeSqrDiff(mKeptSamplesNNTransformed, mKeptSamples);
        aSqrDiff[bunPicked] = dSmolBunToBigBun;




    }
    cout << "Number of pt matrix saved: " << iterations.size() << endl;
    viewer.data().clear();
    for(int nBunId = 0; nBunId < vBunPts.size(); nBunId++){
        viewer.data().add_points(vBunPts[nBunId], vColours[nBunId]);
    }
    viewer.data().point_size = point_size;
    viewer.callback_key_down =  &key_down;
    viewer.launch();

    return 0;
}
