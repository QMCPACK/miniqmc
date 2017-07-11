#include <iostream>
#include <vector>
#include <mpi/mpi.hpp>
using namespace qmcplusplus;
using namespace std;

int main(int argc, char** argv)
{
  mpi::environment env(argc,argv);
  mpi::communicator comm;

  if(comm.size()%4 == 0)
  {
    mpi::communicator row=comm.split(comm.rank()/4);
    std::cout << "Split " << comm.rank() << ":" << row.rank()  << std::endl;
  }

  int num_s=comm.rank();
  int num_r=-1;
  
  //reduction of an ingeger value
  {
    int sum_i=0;
    int max_i=-1;
    int min_i=0;
    int sum_x=0;

    mpi::all_reduce(comm,num_s,sum_i,std::plus<int>());
    mpi::all_reduce(comm,num_s,sum_x);

    mpi::all_reduce(comm,num_s,max_i,mpi::maximum<int>());
    mpi::all_reduce(comm,num_s,min_i,mpi::minimum<int>());

    if(comm.rank()==0)
    {
      std::cout << "Check allreduce " << sum_i << " " << sum_x << " " << max_i << " " << min_i  <<std::endl;
    }

  }


  if(comm.rank()==0)
    comm.send(1,1111,num_s);
  if(comm.rank()==1)
    comm.recv(0,1111,num_r);

  //test double, array reduction operations
  {
    double v_s[4]={comm.rank(),comm.rank(),comm.rank(),comm.rank()};
    double v_r[4]={-1,-1,-1,-1};
    double v_sum[4]={0.0,0.0,0.0,0.0};

    if(comm.rank()==0)
      comm.send(1,12,v_s,4);
    if(comm.rank()==1)
      comm.recv(0,12,v_r,4);

    mpi::all_reduce(comm,v_s,4,v_sum,std::plus<double>());
    if(comm.rank()==0)
    {
      std::cout << "Check allreduce of an array ";
      for(int i=0; i<4; ++i) std::cout << v_sum[i] << " ";
      std::cout << std::endl;
    }
    for(int i=0; i<4; ++i) v_sum[i]=0.0;

    mpi::reduce(comm,v_s,4,v_sum,std::plus<double>(),0);
    if(comm.rank()==0)
    {
      std::cout << "Check reduce of an array ";
      for(int i=0; i<4; ++i) std::cout << v_sum[i] << " ";
      std::cout << std::endl;
    }
    if(comm.rank()==1)
    {
      std::cout << "Check reduce of an array ";
      for(int i=0; i<4; ++i) std::cout << v_sum[i] << " ";
      std::cout << std::endl;
    }
  }

  //test double, array reduction operations
  {
    double v_s[4]={comm.rank(),comm.rank(),comm.rank(),comm.rank()};
    double v_r[4]={-1,-1,-1,-1};

    mpi::request areq;
    bool sender=(comm.rank()%2==0);
    int des=(sender)? comm.rank()+1:comm.rank();
    int src=(sender)? comm.rank():comm.rank()-1;
    if(sender)
      areq=comm.isend(des,17,v_s,4);
    else
      areq=comm.irecv(src,17,v_r,4);

    areq.wait();

    std::cout << "isend/irecv rank:" << comm.rank();;
    for(int i=0; i<4; ++i) std::cout << " "<< v_r[i] ;
    std::cout << std::endl;
  }

  //test complex
  {
    std::complex<float> c_s=comm.rank();
    std::complex<float> c_r={-1.0f,0.0f};

    if(comm.rank()==1)
      comm.send(2,13,c_s);
    if(comm.rank()==2)
    {
      comm.recv(1,13,c_r);
      std::cout << "BCAST " << comm.rank() << ":" << c_s << c_r << std::endl;
    }
  }

  //test complex
  {
    vector<float> c_s(5,float());
    if(comm.rank()==0)
    {
      std::fill(c_s.begin(),c_s.end(),1.1);
    }

    mpi::broadcast(comm,c_s.data(),c_s.size(),0);

    if(comm.rank()==2)
    {
      std::cout << "BCAST " << comm.rank() << ":" << c_s[2] << std::endl;
    }
  }

  return 0;

}
