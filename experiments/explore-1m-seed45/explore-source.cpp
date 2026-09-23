// Simulator-only novelty exploration. No solver, heuristic distance, or answers.
// Archive returns are simulator resets during TRAINING, never during evaluation.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
using State=uint64_t;
struct Car { int horizontal,length,lane; };
struct Node { State s; int parent,action,depth,cost; };
struct Board {
 std::string name; int n; std::array<Car,14> c; State start=0;
 int pos(State s,int i)const{return (s>>(3*i))&7;}
 std::array<int,36> grid(State s)const {
  std::array<int,36> g;g.fill(-1);
  for(int i=0;i<n;i++)for(int k=0;k<c[i].length;k++){
   int cell=c[i].horizontal?c[i].lane*6+pos(s,i)+k:(pos(s,i)+k)*6+c[i].lane;
   g[cell]=i;
  }return g;
 }
 bool solved(State s)const{
  auto g=grid(s);
  for(int x=pos(s,0)+c[0].length;x<6;x++)if(g[c[0].lane*6+x]!=-1)return false;
  return true;
 }
 std::vector<int> actions(State s)const{
  auto g=grid(s);std::vector<int>a;
  for(int i=0;i<n;i++)for(int dir=0;dir<2;dir++)for(int d=1;d<=5;d++){
   int tip=dir?pos(s,i)+c[i].length-1+d:pos(s,i)-d;
   // No exit action is needed: clearance terminates the episode.
   if(tip<0||tip>=6)break;
   int cell=c[i].horizontal?c[i].lane*6+tip:tip*6+c[i].lane;
   if(g[cell]!=-1)break;
   a.push_back(i*10+dir*5+d-1);
  }return a;
 }
 State move(State s,int a)const{
  int i=a/10,code=a%10,p=pos(s,i)+(code<5?-1:1)*(code%5+1);
  return (s&~(State(7)<<(3*i)))|(State(p)<<(3*i));
 }
};
int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"explore boards.txt output.tsv archive|archive-score|archive-refine|random seed budget refinement_steps\n";return 2;}
 std::ifstream input(argv[1]);std::ofstream out(argv[2]);std::string mode=argv[3];
 std::mt19937 rng(std::stoul(argv[4]));long budget=std::stol(argv[5]);long refinement_steps=std::stol(argv[6]);
 if(!input||!out||(mode!="archive"&&mode!="archive-score"&&mode!="archive-refine"&&mode!="random"))return 3;
 int total=0;input>>total;int successes=0;
 for(int p=0;p<total;p++){
  Board b;input>>b.name>>b.n;
  for(int i=0;i<b.n;i++){int pos;input>>b.c[i].horizontal>>b.c[i].length>>b.c[i].lane>>pos;b.start|=State(pos)<<(3*i);}
  std::vector<Node> nodes={{b.start,-1,-1,0,0}};std::unordered_map<State,int> visited={{b.start,0}};
  std::vector<int> path;long steps=0,resets=0,stop_at=budget;bool done=b.solved(b.start);int best_cost=1000000000;
  while(steps<stop_at&&(!done||mode=="archive-refine")){
   int index=mode!="random"?int(rng()%nodes.size()):0;
   State state=nodes[index].s;std::vector<int> episode;resets++;
   if(b.solved(state))continue;
   int length=mode!="random"?32:300;
   for(int t=0;t<length&&steps<stop_at;t++){
    auto legal=b.actions(state);if(legal.empty())break;
    int a=legal[rng()%legal.size()];State next=b.move(state,a);steps++;episode.push_back(a);
    auto hit=visited.find(next);
    int observed_cost=nodes[index].cost+a%5+1;
    if(hit==visited.end()){
     int j=nodes.size();nodes.push_back({next,index,a,nodes[index].depth+1,observed_cost});visited[next]=j;index=j;
    }else {
     int j=hit->second;
     // Go-Explore-style archive score replacement: preserve a better-return
     // route to a cell ONLY after actually observing the sampled transition.
     // No graph sweep / heuristic / unseen successor backup is performed.
     if((mode=="archive-score"||mode=="archive-refine")&&observed_cost<nodes[j].cost){
      nodes[j].parent=index;nodes[j].action=a;nodes[j].depth=nodes[index].depth+1;nodes[j].cost=observed_cost;
     }
     index=j;
    }
    state=next;
    if(b.solved(state)){
     if(!done&&mode=="archive-refine")stop_at=std::min(budget,steps+refinement_steps);
     done=true;std::vector<int> candidate;
     if(mode=="random")candidate=episode;
     else {for(int j=index;nodes[j].parent!=-1;j=nodes[j].parent)candidate.push_back(nodes[j].action);std::reverse(candidate.begin(),candidate.end());}
     int cost=6-b.pos(state,0)-b.c[0].length+1;for(int action:candidate)cost+=action%5+1;
     if(cost<best_cost){best_cost=cost;path=candidate;}
     break;
    }
   }
  }
  successes+=done;out<<b.name<<'\t'<<done<<'\t'<<steps<<'\t'<<resets<<'\t'<<nodes.size()<<'\t';
  for(auto a:path)out<<a<<',';out<<'\n';out.flush();
  if((p+1)%100==0||p+1==total)std::cout<<mode<<" boards "<<p+1<<" discovered "<<successes<<std::endl;
 }
}
