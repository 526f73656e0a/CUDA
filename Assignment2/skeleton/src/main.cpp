/// /// @file
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : tutorial 5
///
///  project    : GPU Programming
///
///  description: merge sort
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, syste
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>
#include <vector>
#include <cstring>
int glob;
typedef std::chrono::milliseconds TimeT;
////////////////////////////////////////////////////////////////////////////////////////////////////
//! Parallel merge step of merge sort
////////////////////////////////////////////////////////////////////////////////////////////////////
// It works only for lengths up to around 64 elements, after that something in the recursive call of mergeParallel in the if(size1>1&&size2>1) chunk breaks. 
// Because it got up to a pretty lengthy array it is hard for me to find the exact mistake, so i'm leaving it so, it is probably something really small and stupid  
template<typename T>
void
mergeParallel(T* data1, T* last1, T* data2, T* last2, T* scratch,
    int num_threads) {
    int size1 = last1 - data1;
    int size2 = last2 - data2;
    int size = size1 + size2;
    //std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //std::cout << "=======================This is mergesort call # " << glob << "=======================" << std::endl;
    //glob++;
    for (int i = 0; i < size; ++i) {
        //std::cout << "Entry data [i] = " << data1[i] << std::endl;
    }
   // std::cout << "======================= Sizes are =======================" << std::endl;
    //std::cout
        //<< "size1 = " << size1 << std::endl
        //<< "size2 = " << size2 << std::endl
        //<< "size = " << size << std::endl;

    if (size1 > 1 && size2 > 1) {
        int pivot1 = (size1 / 2);
        if (size1 % 2 != 0)pivot1 = (size1 + 1) / 2;
        int pivot2;
        if (data1[pivot1] > data2[size2 - 1]) {
           // std::cout << "===============Pivot==================" << std::endl;
           // std::cout << "data1[pivot] = " << data1[pivot1] << std::endl;
            int j = 0;
            for (int i = 0; i < pivot1; ++i) {
                scratch[j] = data1[i]; ++j;
            }
            for (int i = 0; i < size2; ++i) {
                scratch[j] = data2[i]; ++j;
            }
            for (int i = pivot1; i < size1; ++i) {
                scratch[j] = data1[i]; ++j;
            }
            for (int i = 0; i < size; ++i) {
                data1[i] = scratch[i];
            }
           // std::cout << "===============Printing rearanged data==================" << std::endl;
            for (int i = 0; i < size; ++i) {
                data1[i] = scratch[i];
               // std::cout << "changed data1[i] = " << data1[i] << std::endl;
            }
            //int size3 = size1 - pivot1;
            //int mid = size3 / 2 + size3 % 2 + pivot1+size2;

            // the array A2 is already sorted so we just call the func for A1 and B1 
            mergeParallel(data1, data1 + pivot1, data1 + pivot1, data1 + pivot1 + size2, scratch, num_threads);
            //mergeParallel(data1+pivot1+size2,data1+mid,data1+mid,last2,scratch+pivot1+size2,num_threads);

        }
        else if (data1[pivot1] < data2[0]) {
            return;
            if (data1[size1 - 1] < data2[0]) {
                return;
            }
            else {
                if (data1[size1 - 1] < data2[0])return;
                for (int i = pivot1; i < size1; ++i) {
                    for (int j = size2 - 1; j <= 0; --j) {
                        if (data1[i] < data2[j]) { pivot1 = i; pivot2 = j; break; }
                    }
                }

               // std::cout << "===============Pivot==================" << std::endl;
               // std::cout << "data1[pivot] = " << data1[pivot1] << std::endl;
                int j = 0;
                for (int i = 0; i < pivot1; ++i) {
                    scratch[j] = data1[i]; ++j;
                }
                for (int i = 0; i < pivot2; ++i) {
                    scratch[j] = data2[i]; ++j;
                }
                for (int i = pivot1; i < size1; ++i) {
                    scratch[j] = data1[i]; ++j;
                }
                for (int i = pivot2; i < size2; ++i) {
                    scratch[j] = data2[i]; ++j;
                }
                for (int i = 0; i < size; ++i) {
                    data1[i] = scratch[i];
                   // std::cout << "changed data1[i] = " << data1[i] << std::endl;
                }
            }
        }
        else {

            for (int i = 0; i < size2; ++i) {
                if (data1[pivot1] < data2[i]) { pivot2 = i; break; }
            }
           // std::cout << "===============Pivot==================" << std::endl;
           // std::cout << "pivot1 = " << pivot1 << std::endl;
           // std::cout << "pivot2 = " << pivot2 << std::endl;
           // std::cout << "data1[pivot] = " << data1[pivot1] << std::endl;
            int j = 0;
            for (int i = 0; i < pivot1; ++i) {
                scratch[j] = data1[i]; ++j;
            }
            for (int i = 0; i < pivot2; ++i) {
                scratch[j] = data2[i]; ++j;
            }
            for (int i = pivot1; i < size1; ++i) {
                scratch[j] = data1[i]; ++j;
            }
            for (int i = pivot2; i < size2; ++i) {
                scratch[j] = data2[i]; ++j;
            }

           // std::cout << "===============Printing rearanged data==================" << std::endl;
            for (int i = 0; i < size; ++i) {
                data1[i] = scratch[i];
            //    std::cout << "changed data1[i] = " << data1[i] << std::endl;
            }
            int left1, left2;
            left1 = size1 - pivot1;
            left2 = size2 - pivot2;
            if (num_threads > 1) {
                auto t = std::thread(mergeParallel<T>, data1, data1 + pivot1, data1 + pivot1, data1 + pivot1 + pivot2, scratch, num_threads / 2 + num_threads % 2);
                mergeParallel(data1 + pivot1 + pivot2, data1 + pivot1 + pivot2 + left1, data1 + pivot1 + pivot2 + left1, last2, scratch + pivot1 + pivot2, num_threads/2);
                t.join();
            }
            else {
                mergeParallel(data1, data1 + pivot1, data1 + pivot1, data1 + pivot1 + pivot2, scratch, num_threads);
                mergeParallel(data1 + pivot1 + pivot2, data1 + pivot1 + pivot2 + left1, data1 + pivot1 + pivot2 + left1, last2, scratch + pivot1 + pivot2, num_threads);
            }
        }
    }
    // merge for less elements
    else {
        //std::cout << "Entering merge" << std::endl;
        if (size1 == 0) {
            merge(data2, last2, size2 / 2, scratch);
            for (int i = 0; i < size; ++i) {
               //std::cout << "#####################################################" << std::endl;
               // std::cout << "This is merged data[i]" << data2[i] << std::endl;
            }
        }
        else if (size2 == 0) {
            merge(data1, last1, size1 / 2, scratch);
            for (int i = 0; i < size; ++i) {
               // std::cout << "This is merged data[i]" << data1[i] << std::endl;
            }
        }

        else if (size1 == 1 && size2 == 1) {
            if (data2 [0] < data1 [0]) {
                scratch[0] = data2[0];
                data2[0] = data1[0];
                data1[0] = scratch[0];
                for (int i = 0; i < size; ++i) {
                    //std::cout << "This is merged data[i]" << data1[i] << std::endl;
                }
            }
            else {
                for (int i = 0; i < size; ++i) {
                   // std::cout << "This is merged data[i]" << data1[i] << std::endl;
                }
            }
        }
        else if (size1 > 1 && size2 == 1) {
            int i;
            int k = 0;
            if (data2[0] > data1[size1 - 1]) {
                for (int j = 0; j < size; ++j) {
                    //std::cout << "This is merged data[i]" << data1[j] << std::endl;
                }
                return;
            }
            else{
                for (i = 0; i < size1; ++i) {
                    if (data2[0] < data1[i]) {
                        scratch[k] = data2[0]; ++k; break;
                    }
                    else {
                        scratch[k] = data1[i]; ++k;
                    }
                }
                for (int j = i; j < size1; ++j) {
                    scratch[k] = data1[j]; ++k;
                }
                for (int j = 0; j < size; ++j) {
                    data1[j] = scratch[j];
                    //std::cout << "This is merged data[i]" << data1[j] << std::endl;
                }
            }
            //merge(data1, last2, size / 2, scratch);

        }
        else if (size1 == 1 && size2 > 1) {
            int i;
            int k = 0;
            if (data1[0] > data2[size2 - 1]) {
                for (i = 0; i < size2; ++i) {
                    scratch[k] = data2[i]; ++k;
                }
                std::cout << data1[0] << std::endl;
                scratch[k] = data1[0];
                for (int j = 0; j < size; ++j) {
                    data1[j] = scratch[j];
                    //std::cout << "This is merged data[i]" << data1[j] << std::endl;
                }
                return;
            }
            else {
                for (i = 0; i < size2; ++i) {

                    if (data1[0] < data2[i]) {
                        scratch[k] = data1[0]; ++k;break;
                    }
                    else { scratch[k] = data2[i]; ++k; }
                }
                for (int j = i; j < size2; ++j) {
                    scratch[k] = data2[j]; ++k;
                }
                for (int j = 0; j < size; ++j) {
                    data1[j] = scratch[j];
                    //std::cout << "This is merged data[i]" << data1[j] << std::endl;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Merge sort
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
void
mergesort(T* data, T* last, T* scratch) {
    // termination
    int size = last - data;
    // recursion
    if (size >= 2) {
        int mid = size / 2;
        int ls = mid;
        if (size % 2 != 0) {
            ls++;
        }
        mergesort(data, data + ls, scratch);
        mergesort(data + ls, last, scratch + ls);
        //std::cout << "Starting Merge" << std::endl;
        merge(data, last, mid, scratch);
    }
    else return;

    // merging
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//! Merge step of merge sort
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
void
merge(T* data, T* last, int nhalf, T* scratch) {
    //serial merge step
    int size = last - data;
    int ls = nhalf;
    if (size % 2 != 0) {
        ls++;
    }
    int i, j, k;
    i = 0; j = ls; k = 0;
    while (i < ls && j < size) {
        if (data[i] <= data[j]) {
            scratch[k] = data[i];
            i++;
            k++;
        }
        else {
            scratch[k] = data[j];
            j++;
            k++;
        }
    }
    while (i < ls) {
        scratch[k] = data[i];
        i++;
        k++;
    }
    while (j < size) {
        scratch[k] = data[j];
        j++;
        k++;
    }
    for (int p = 0; p < size; p++) {
        data[p] = scratch[p];
        //scratch[p] = 0;
    }

}
////////////////////////////////////////////////////////////////////////////////////////////////////
//! Merge sort
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
void
mergesortParallel(T* data, T* last, T* scratch, const int num_threads) {
    int size = last - data;
    //std::cout << "Data Splitting" << std::endl;
    //for (int i = 0; i < size; i++) {
      //  std::cout << "Data[" << i << "] is = " << data[i] << std::endl;}
   
#if 1
    // serial merge step
    if (size >= 2) {
        int mid = size / 2;
        int ls = mid;
        if (size % 2 != 0) {
            ls++;
        }
        if (num_threads > 1) {
            //std::cout << "Creating new thread" << std::endl;
            //std::cout << "Thread 1 works from element:"<<*data <<"to element:"<<*(data+ls) << std::endl;
            auto t = std::thread(mergesortParallel<T>, data, data + ls, scratch, (num_threads / 2 + num_threads % 2));
            //std::cout << "Thread 0 works from element:" <<*( data+ls) << "to element:" << *(last-1) << std::endl;
            mergesortParallel(data + ls, last, scratch + ls, num_threads / 2);
            t.join();
        }
        else {
            //std::cout << "Works from element:" << *data << "to element:" << *(data + ls) << std::endl;

            mergesortParallel(data, data + ls, scratch, num_threads);

            //std::cout << "Works from element:" << *(data + ls) << "to element:" << *(last - 1) << std::endl;

            mergesortParallel(data + ls, last, scratch + ls, num_threads);
            //std::cout << "Starting Merge" << std::endl;
            merge(data, last, mid, scratch);
        }
}
    else return;
    merge(data, last, size / 2, scratch);
   

#else
    if (size >= 2) {
        int mid = size / 2;
        int ls = mid;
        if (size % 2 != 0) {
            ls++;
        }
        // save some threads for the mergeParallel function 
        if (num_threads > 4) {
            //std::cout << "Creating new thread" << std::endl;
            //std::cout << "Thread 1 works from element:"<<*data <<"to element:"<<*(data+ls) << std::endl;
            auto t = std::thread(mergesortParallel<T>, data, data + ls, scratch, (num_threads / 2 + num_threads % 2));
            //std::cout << "Thread 0 works from element:" <<*( data+ls) << "to element:" << *(last-1) << std::endl;
            mergesortParallel(data + ls, last, scratch + ls, num_threads / 2);
            t.join();


        }
        else {
            //std::cout << "Works from element:" << *data << "to element:" << *(data + ls) << std::endl;

            mergesortParallel(data, data + ls, scratch, num_threads);

            //std::cout << "Works from element:" << *(data + ls) << "to element:" << *(last - 1) << std::endl;

            mergesortParallel(data + ls, last, scratch + ls, num_threads);
        }
       //std::cout << "Starting Merge" << std::endl;

       mergeParallel(data, data + ls, data + ls, last, scratch, num_threads);


    }
    else return;

#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// program entry point
////////////////////////////////////////////////////////////////////////////////////////////////////
int
main(int /*argc*/, char** /*argv*/) {

    int rep = 1000; // how many repetitions 
    int teststart = 0;
    int testend = 1;
    // TEST for sizes 2^k , where k is 8,9,10,11,12,13,14 ..
    for(int exponent = teststart;exponent<testend;++exponent ){
    // int n = pow(2,exponent);
    int n = 1048578/8; //2^20

    std::cout << "exponent = " << exponent << std::endl;

    int* data = (int*)malloc(n * sizeof(int));
    int* scratch = (int*)malloc(n * sizeof(int));
    int* dataref = (int*)malloc(n * sizeof(int));

    bool correct = true;
    const unsigned int num_threads = std::thread::hardware_concurrency();
    std::cout << "num_threads = " << num_threads << std::endl;

    //Basic Test
    /*
    data[0] = 5;
    data[1] = 4;
    data[2] = 12;
    data[3] = 2;
    data[4] = 7;
    data[5] = 1;
    data[6] = 9;
    data[7] = 8;
    data[8] = 3;
    data[9] = 10;
    data[10] = 13;
    data[11] = 6;
    /**/
    //Execute Mergesort 
    // Benchmark


    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < rep; i++)
    {
        //Random data for each iteration 
        std::generate(data, data + n, std::rand);
        memcpy(dataref, data, sizeof(int) * n);

        // MERGESORT
        //mergesort(data, data + n, scratch);

        // MERGESORT PARALLEL 
        mergesortParallel(data, data + n, scratch, num_threads);

        // Divide And Conquer 
        //mergesortParallel(data, data + n, scratch, 1);

        //STD::SORT
        //std::sort(dataref, dataref + n);
    }
    double time = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start).count();
    // compare to the reference

    std::sort(dataref, dataref + n);

    for (int i = 0; i < n; ++i) {
        correct &= (data[i] == dataref[i]);
        //std::cout << "data[i]=" << data[i] << std::endl;
    }

    // print out the time and if it is correct. 
    std::cout << "Std Sort " << ((correct) ? "succeeded." : "failed.")
        << " It took " << time
        << "ms." << std::endl;

    // clean up
    free(data);
    free(scratch);
    free(dataref);
    }
    return EXIT_SUCCESS;
}
