#include "graph.h"
#include "common_ops.h"
#include "common_type.h"
#include "api.h"

#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <string>

// ============================================================
// Adopted directly from the original code
// Data structure for Pilos
// ============================================================
class CSV
{
public:
    VertexID ID;
    std::vector<std::pair<VertexID, VertexID>> edges;
    int *Nedge{nullptr};
    bool NedgeC = false;
    bool change = true;
    bool Ichange = true;
    bool IPchange = true;
    bool deleted = false;

public:
    CSV(int eigens, VertexID IDV, uint32_t maxDeg)
    {
        ID = IDV;
        deleted = false;
        change = true;
        Ichange = true;
    }

    CSV(uint32_t IDV)
    {
        ID = IDV;
        deleted = false;
        change = true;
        Ichange = true;
    }

    CSV()
    {
        ID = -1;
    }
};

// ============================================================
// Adopted directly from the original code
// Egien Value Helper Function
// ============================================================

inline VertexID checkANX(std::vector<VertexID> ID, VertexID CID)
{
    for (int i = 0; i < ID.size(); i++)
        if (ID[i] == CID)
            return i;
    return INVALID;
}

void calcEigens1(Eigen::SparseMatrix<double> M, int k, Eigen::VectorXd &evalues, int count)
{
    using namespace Eigen;

    int sizek = k * 2;
    int dek = k;
    //
    // M.makeCompressed();
    // if(!M.isApprox(M.transpose()))
    // cout<<"really?"<<endl;

    SelfAdjointEigenSolver<SparseMatrix<double>> eigensolver(M);

    // EigenSolver<SparseMatrix<double>> eigensolver(M);
    if (eigensolver.info() != Success)
    {
        std::cerr << "Eigenvalue computation failed!" << std::endl;
        return;
    }
    VectorXd eigenvalues = eigensolver.eigenvalues();

    if (eigenvalues.size() < k)
    {
        uint32_t ts = eigenvalues.size();
        evalues = eigenvalues.tail(eigenvalues.size()).reverse();
        // cout<<evalues<<endl;
        // cout<<"count "<<count<<" NE "<<evalues.size();
        int sz = evalues.size();
        evalues.conservativeResize(dek);
        if (ts == count)
            evalues(sz) = -1;
        else
            evalues(sz) = 0;
        sz++;
        for (int i = sz; i < dek; i++)
            evalues(i) = -1;
    }
    else
    {
        evalues = eigenvalues.tail(k).reverse();
    }
}

void CompactADLEIG(const Graph& data, int degree, Eigen::VectorXd &evalues, VertexID vertex, int depth, int Eprun, int oMax)
{
    typedef Eigen::Triplet<double> T;
    using namespace Eigen;
    using namespace std;

    vector<T> tripletList;
    vector<VertexID> ID; // add size then resize
    vector<VertexID> IDD;
    vector<VertexID> IDL;
    VertexID *neighbors_;
    VertexID vx1;
    VertexID vx2;
    VertexID vertexpair;
    VertexID vertexprint;
    vertexprint = vertex;
    int count = 0;
    uint32_t u_nbrs_count;
    uint32_t u_nbrs_count1;
    const VertexID *u_nbrs = data.getNeb(vertex, u_nbrs_count);
    int k = Eprun;
    if (u_nbrs_count > oMax)
    {
        evalues.resize(k);
        for (int i = 0; i < k; i++)
            evalues(i) = 500;
        return;
    }
    ID.push_back(vertex);
    IDL.push_back(1);
    queue<VertexID> q_curr;
    queue<VertexID> q_next;
    queue<VertexID> q_temp;
    tripletList.push_back(T(0, 0, (float)u_nbrs_count));
    for (int j = 0; j < u_nbrs_count; ++j)
    {
       // if (u_nbrs[j] == vertex)
        //    cout << "error :" << vertex << endl;
        count++;
        tripletList.push_back(T(0, count, -1));
        tripletList.push_back(T(count, 0, -1));
        ID.push_back(u_nbrs[j]);
        IDL.push_back(1);
        q_curr.push(u_nbrs[j]);
    }

    for (int i = 1; i < depth; i++)
    {
        while (!q_curr.empty())
        {
            vertex = q_curr.front();
            q_curr.pop();
            u_nbrs = data.getNeb(vertex, u_nbrs_count);
            if (u_nbrs_count > oMax)
            {
                evalues.resize(k);
                for (int tt = 0; tt < k; tt++)
                    evalues(tt) = 500;
                return;
            }
            vx1 = checkANX(ID, vertex);
            tripletList.push_back(T(vx1, vx1, (float)u_nbrs_count));
            for (uint32_t j = 0; j < u_nbrs_count; ++j)
            {

                vertexpair = u_nbrs[j];
                if (vertexpair == vertex)
                    cout << "problem again!!" << endl;

                vx2 = checkANX(ID, vertexpair);
                if (vx2 == INVALID)
                {
                    count++;
                    vx2 = count;
                    q_next.push(vertexpair);
                    ID.push_back(vertexpair);
                    IDL.push_back(1);
                    tripletList.push_back(T(vx1, vx2, -1));
                    tripletList.push_back(T(vx2, vx1, -1));
                    if (i == depth - 1)
                    {
                        IDL[vx2] = 1;
                    }
            if (count > oMax)
            {
                evalues.resize(k);
                for (int tt = 0; tt < k; tt++)
                    evalues(tt) = 500;
                return;
            }
                }
                else
                {
                    tripletList.push_back(T(vx1, vx2, -1));
                    tripletList.push_back(T(vx2, vx1, -1));
                    if (i == depth - 1)
                    {
                        IDL[vx2]++;
                    }
                }
            }
        }
        if (!q_next.empty())
            q_curr.swap(q_next);
        else
        {
            i = depth;
        }
    }
    while (!q_curr.empty())
    {
        vertex = q_curr.front();
        q_curr.pop();
        vx1 = checkANX(ID, vertex);
        tripletList.push_back(T(vx1, vx1, IDL[vx1]));
    }
    count++;
    map<int, int> count_uniques;
    set<pair<int, int>> seen;
    vector<Triplet<double>> unique_triplets;
    for (auto t : tripletList)
    {
        if (seen.count({t.row(), t.col()}) == 0)
        {
            unique_triplets.push_back(Triplet<double>(t.row(), t.col(), t.value()));
            seen.insert({t.row(), t.col()});
            count_uniques[t.row()]++;
        }
    }
    tripletList = unique_triplets;
    if (tripletList.size() == count * count)
    {
        // cout << "MEga prob" << endl;
        evalues.resize(k);

        for (int ss = 0; ss < k; ss++)
        {
            if (ss < count)
                // evalues(ss)=count-1;
                evalues(ss) = count;
            else if (ss == count)
                evalues(ss) = 0;
            else // evalues(ss)=-1;
                evalues(ss) = 0;
        }
    }
    else
    {
        SparseMatrix<double> M(count, count);
        M.setFromTriplets(tripletList.begin(), tripletList.end(), [](double a, double b)
                          { return b; });
        // checkM(M);

       calcEigens1(M, k, evalues, count);
        //calcEigens12(M,k,evalues,count);
    }
    tripletList.clear();
    ID.clear();
    IDL.clear();
}

void MTcalc12(const Graph& data, int degree, Eigen::MatrixXd &eigenVD, bool LE, int Eprun, int oMax)
{

    Eigen::VectorXd evalues;
    int siz = data.getVertexCnt();
    for (int i = 0; i < siz; i++)
    {
        CompactADLEIG(data, degree, evalues, i, 2, Eprun, oMax);
        eigenVD.row(i) = evalues;
        evalues.setZero();
    }

}

// ============================================================
// Adopted directly from the original code
// PILOS filter helper function
// ============================================================

/*Extract Label Information for the query
 *Createas a map with Label ID and count that we can easily compare
 *with a candidate node
 */
void ExtractNImap(const Query& query, std::vector<std::map<LabelID, int>> &QueryNlabel)
{
    const VertexID *u_nbrs;

    uint32_t u_nbrs_count;
    uint32_t qsiz = query.getVertexCnt();
    for (int i = 0; i < qsiz; i++)
    {
        u_nbrs = query.getNeb(i, u_nbrs_count);
        std::map<VertexID, int> QueryVec;
        for (int j = 0; j < u_nbrs_count; j++)
        {
            LabelID labela = query.getVertexLabel(u_nbrs[j]);
            if (QueryVec.count(labela) == 0)
            {
                // Key does not exist, add it with a value of 1
                QueryVec[labela] = 1;
            }
            else
            {
                // Key exists, increment its value
                QueryVec[labela]++;
            }
        }
        QueryNlabel.emplace_back(QueryVec);
        QueryVec.clear();
    }
}


/*Extract Label Information for the query for 2hops
 *Createas a map with Label ID and count that we can easily compare
 *with a candidate node. Self Included!
 */
void ExtractUI2h(const Graph& query, std::vector<uint32_t> &Deg, std::vector<std::map<uint32_t, int>> &QueryNlabel2, int **&VS)
{
    using namespace std;

    const VertexID *u_nbrs;
    uint32_t u_nbrs_count;
    const VertexID *u_nbrsN;
    uint32_t u_nbrs_countN;
    set<uint32_t> QueryVec;
    map<uint32_t, int> QueryVec1;
    uint32_t labela = 0;
    uint32_t qsiz = query.getVertexCnt();

    // int matrix[qsiz][qsiz] = {-1};
    int **matrix = new int *[qsiz];
    for (int i = 0; i < qsiz; i++){
        matrix[i] = new int[qsiz];
        for (int j = 0; j < qsiz; j++){
            matrix[i][j] = -1;
        }
    }
    
    for (int i = 0; i < qsiz; i++)
    {
        QueryVec.insert(i);
        u_nbrs = query.getNeb(i, u_nbrs_count);

        for (int j = 0; j < u_nbrs_count; j++)
        { // First hop
            if (QueryVec.insert(u_nbrs[j]).second)
            {
                matrix[i][u_nbrs[j]] = i;
                matrix[u_nbrs[j]][i] = i;
                labela = query.getVertexLabel(u_nbrs[j]);
                if (QueryVec1.count(labela) == 0)
                {
                    // Key does not exist, add it with a value of 1
                    QueryVec1[labela] = 1;
                }
                else
                {
                    // Key exists, increment its value
                    QueryVec1[labela]++;
                }
            }
            // second Hop
            u_nbrsN = query.getNeb(u_nbrs[j], u_nbrs_countN);
            for (int k = 0; k < u_nbrs_countN; k++)
            {
                matrix[u_nbrsN[k]][u_nbrs[j]] = i;
                matrix[u_nbrs[j]][u_nbrsN[k]] = i;
                if (u_nbrsN[k] == i)
                    continue;
                labela = query.getVertexLabel(u_nbrsN[k]);
                if (QueryVec.insert(u_nbrsN[k]).second)
                {
                    if (QueryVec1.count(labela) == 0)
                    {
                        // Key does not exist, add it with a value of 1
                        QueryVec1[labela] = 1;
                    }
                    else
                    {
                        // Key exists, increment its value
                        QueryVec1[labela]++;
                    }
                }
            }
        }
        int sumr = 0;
        for (int dd = 0; dd < qsiz; dd++)
        {
            sumr = 0;
            for (int kk = 0; kk < qsiz; kk++)
            {
                if (matrix[dd][kk] == i)
                    sumr++;
            }
            VS[i][dd] = sumr;
        }
        Deg.emplace_back(QueryVec.size());
        QueryVec.clear();
        QueryNlabel2.emplace_back(QueryVec1);
        QueryVec1.clear();

        std::sort(VS[i], VS[i] + qsiz, std::greater<int>());
    }

    for (int i = 0; i < qsiz; i++)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}

/* Add vertices that Pass NLF and Eigen rule to candidate Space.
 * OpenData1 need to be removed from here and pass the eigenVD1 as parameter(index)
 **310 is hardcodes max number of label ID-> can be just extracted from data graph->number of Labels.
 **I preload for every label all the possible candidate nodes, so I will not have to reload them if
 **a query node has the same label.
 **Important to note by construction the nodes are ordered by ID so FCS[0][0].ID<FCS[0][1].ID
 **1)Pruning 1 l(q)=l(v) So labelsNum[reverseLab[label]] already has nodes that have the same label
 **2)Pruning 2 d(q)<=d(v) Degree
 **3)Pruning 3 LE(q)<=LE(v) top S laplacian
 **4)Pruning 4 N(q)<=N(v) Neigboorhood labels
 */

void Vertices(std::vector<std::vector<CSV>> &FCS, const Graph& data, const Query& query, float **&eigenVq1, std::vector<std::map<uint32_t, int>> &QueryNlabel, float **&eigenVD1)
{
    using namespace std;
    using namespace Eigen;

    int qsiz = query.getVertexCnt();
    int dsiz = data.getVertexCnt();

    VectorXd devalues;
    VectorXd qevalues;
    bool con = true;
    uint32_t com = data.getMaxLabelFreq();
    uint32_t copies = query.getLabelCnt() + 1;
    uint32_t labelsNum[copies];
    const uint32_t *labelData[copies];
    uint32_t kk;
    uint32_t i;
    uint32_t j;
    uint32_t reverseLab[310];
    uint32_t u_nbrs_countD = 0;

    LabelID label = 0;
    for (i = 0; i < 310; i++)
        reverseLab[i] = 310;
    int pos = 0;
    for (i = 0; i < qsiz; i++)
    {
        label = query.getVertexLabel(i);
        uint32_t vdata_vertex_num = 0;
        if (reverseLab[label] == 310)
        {
            reverseLab[label] = pos;
            labelData[pos] = data.getVertexByLabel(label, vdata_vertex_num);
            labelsNum[pos] = vdata_vertex_num;
            pos++;
        }
    }
    uint32_t reserveS;
    vector<CSV> CS;
    CS.reserve(com);
    FCS.reserve(qsiz);
    uint32_t k = 0;
    uint32_t degree = 0;
    uint32_t data_vertex_num;
    int prunES = 30;
    if (prunES>qsiz){
        prunES = qsiz-1;
    }
        
    // for every C(q)
    uint32_t vdata_vertex_num = 0;
    for (i = 0; i < qsiz; i++)
    { 
        
        label = query.getVertexLabel(i);
        degree = query.getVertexDegree(i);
        data_vertex_num = 0;
        for (j = 0; j < labelsNum[reverseLab[label]]; ++j)
        {
            // get Vertex ID
            VertexID data_vertex = labelData[reverseLab[label]][j];
            if (data.getVertexDegree(data_vertex) >= degree)
            {
                con = true;
                //prunES=0;
                // Eigen Value Pruning up to pruneEs value
                //prunES=0;
                #ifdef EIGEN_INDEX
                for (kk = 0; kk < prunES; kk++)
                {
                    if (eigenVq1[i][kk] <0)
                        break;
                    if (eigenVD1[data_vertex][kk] < eigenVq1[i][kk])
                    //if (eigenVD1[data_vertex][kk] - eigenVq1[i][kk]<-0.0001)
                        // Rounding errors for eigenvalue
                        if ((eigenVq1[i][kk] - eigenVD1[data_vertex][kk]) > 0.0001)
                        {
                            con = false;
                            break;
                        }
                }
                #endif

                if (con)
                {
                    // Neigborhood Check
                    for (auto it = QueryNlabel[i].begin(); it != QueryNlabel[i].end(); ++it)
                    {
                        data.getNebByLabel(data_vertex, u_nbrs_countD, it->first);
                        if (u_nbrs_countD < it->second)
                        {
                            con = false;
                            break;
                        }
                    }
                    // If all rules true -> add to candidate space
                    if (con)
                    {
                        CSV cat(data_vertex);
                        CS.emplace_back(cat);
                    }
                }
            }
        }
        FCS.emplace_back(CS);
        CS.clear();
    }
} 

 /*Add edges to the candidate space based on the paper rules.
 */

void EdgesCSBasicRL(std::vector<std::vector<CSV>> &FCS, const Graph& data, const Query& query, uint32_t *&flag, uint32_t *&updated_flag)
{
    int qsiz = query.getVertexCnt();

    uint32_t u_nbrs_count = 0;
    uint32_t u_nbrs_countD = 0;
    int sizA = 0;
    int sizC;
    VertexID VID = 0;
    VertexID de = 0;
    VertexID cne = 0;
    VertexID labela = 0;
    // for CS. for every node of the CS(i)
    for (VertexID a = 0; a < qsiz; a++)
    { // take the neiboors of the query node FCS[a]
        const VertexID *u_nbrs = query.getNeb(a, u_nbrs_count);
        // VertexID* u_nbrs = query_graph->getVertexNeighbors(a, u_nbrs_count);
        sizA = FCS[a].size();
        // Now for every node of the neigboors of FCS[a]---
        for (VertexID c = 0; c < u_nbrs_count; c++)
        { // we start checking query nodes with smaller id to higher
            // thus is the query neigboor has smaller ID we already
            // added the edge to the CS.
            uint32_t updated_flag_count = 0;
            if (u_nbrs[c] < a)
                continue;
            for (int nn = 0; nn < FCS[u_nbrs[c]].size(); nn++)
            {
                flag[FCS[u_nbrs[c]][nn].ID] = nn + 1;
                updated_flag[updated_flag_count++] = FCS[u_nbrs[c]][nn].ID;
            }

            cne = u_nbrs[c];
            // For every node of the CS(i)-> the candidates query node we evaluate
            // candidate vertex for a query a is FCS[a][b].ID
            for (VertexID b = 0; b < sizA; b++)
            {
                VID = FCS[a][b].ID;
                // const VertexID* u_nbrsD = data_graph->getVertexNeighbors(FCS[a][b].ID, u_nbrs_countD); //real neigboors of the candidate vertex
                // get all the neigbors of the FCS[a][b].ID in the data graph
                const VertexID *u_nbrsD = data.getNeb(FCS[a][b].ID, u_nbrs_countD); // real neigboors of the candidate vertex
                                                                                                       // for every neigboor of the candidate vertex of the real graph
                                                                                                       // check if the node exists in the set of neigboors
                for (VertexID e = 0; e < u_nbrs_countD; e++)
                {
                    uint32_t NID = u_nbrsD[e];
                    if (flag[NID] != 0)
                    {
                        FCS[a][b].edges.emplace_back(std::make_pair(cne, NID));
                        FCS[cne][flag[NID] - 1].edges.emplace_back(std::make_pair(a, FCS[a][b].ID));
                    }
                }
            }
            for (uint32_t aa = 0; aa < updated_flag_count; ++aa)
            {
                flag[updated_flag[aa]] = 0;
            }
        }
    }
}


/*Assuming FCS[i][j].ID is sorted we cand find any j given IDC=i and IDQ=FCS[i][j].ID with Binary search.
**Keep in mind sorted properties have to be kept when removing elements.
*/
inline VertexID findIndBS(std::vector<std::vector<CSV>> &FCS, VertexID IDC, VertexID IDQ)
{
    int lo = 0, hi = FCS[IDQ].size() - 1;
    int mid;
    // This below check covers all cases , so need to check
    while (hi - lo > 1)
    {
        int mid = (hi + lo) / 2;
        if (FCS[IDQ][mid].ID < IDC)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    if (FCS[IDQ][lo].ID == IDC)
    {
        return lo;
    }
    else if (FCS[IDQ][hi].ID == IDC)
    {
        return hi;
    }
    std::cout << "error Prob" << std::endl;
    std::cout << IDC << "IDC,IDQ" << IDQ << std::endl;
    return INVALID;
}

/*Initial Pruning. After the CS creation we remove nodes that have
 *less edges that their CS.
 */
bool InitPrunTCSR(std::vector<std::vector<CSV>> &FCS, const Query& query)
{
    int qsiz = query.getVertexCnt();

    int jj = 0;
    uint32_t VDP;
    VertexID i = 0;
    VertexID rev;
    bool ret = false;
    for (VertexID kk = 0; kk < qsiz; kk++)
    {
        jj = FCS[kk].size();
        VDP = query.getVertexDegree(kk);
        while (jj > 0)
        {
            jj--;
            if (FCS[kk][jj].Ichange == true)
            {
                if (FCS[kk][jj].edges.size() == 0)
                {
                    ret = true;
                    FCS[kk][jj].deleted = true;
                }
                // pruning rule
                else if (FCS[kk][jj].edges.size() < VDP)
                {
                    i = 0;

                    while (i < FCS[kk][jj].edges.size())
                    {
                        if (FCS[kk][jj].edges[i].first == INVALID)
                        {
                            i++;
                            continue;
                        }
                        rev = findIndBS(FCS, FCS[kk][jj].edges[i].second, FCS[kk][jj].edges[i].first); // vertex to remove ID?
                        assert(rev != INVALID);
                        FCS[FCS[kk][jj].edges[i].first][rev].Ichange = true;
                        for (int dd = 0; dd < FCS[FCS[kk][jj].edges[i].first][rev].edges.size(); dd++)
                            if (FCS[FCS[kk][jj].edges[i].first][rev].edges[dd].first == kk && FCS[FCS[kk][jj].edges[i].first][rev].edges[dd].second == FCS[kk][jj].ID)
                            {
                                FCS[FCS[kk][jj].edges[i].first][rev].edges[dd].first = INVALID;
                                i++;
                                break;
                            }
                    }
                    FCS[kk][jj].deleted = true;
                    ret = true;
                }

                FCS[kk][jj].Ichange = false;
            }
        }
    }

    return ret;
}

/*Removed all nodes from the Candidate space that are set to be pruned.
**While iterating we set edges to INVALID as a max value
**and notes to csv deleted. Then we gather all the nodes and edges to
** remove them all together to avoid reallocations and reorderings
// ! CHANGED SENTINEL VALUE TO INVALID
*/
inline void clearWrong(std::vector<std::vector<CSV>> &FCS)
{
    for (auto &row : FCS)
    {
        row.erase(remove_if(row.begin(), row.end(),
        [&](CSV &csv){
            if (csv.deleted) {
                return true;
            }
            auto newEnd = remove_if(csv.edges.begin(), csv.edges.end(), [](const std::pair<VertexID, VertexID> &edge) {
                return edge.first == INVALID;
            });
            csv.edges.erase(newEnd, csv.edges.end());
            return false;
        }), 
        row.end());
    }
}

//for memory less requirements check map solution
/*Degree check for every node so we avoid computations in the future.
**Assuming that to be here the degree check is valid
**For every query vertex the node has to have at least 1 edge in
**a neigborhood candidate space. We start by adding all the count
**for every edge and every time we remove 1 edge we decrease the count.
**
**
*/
void fillEN(std::vector<std::vector<CSV>> &FCS, const Query& query)
{
    int qsiz = query.getVertexCnt();

    for (int i = 0; i < qsiz; i++)
    {
        uint32_t de = query.getVertexDegree(i);

        for (int j = 0; j < FCS[i].size(); j++)
        {
            FCS[i][j].Nedge = new int[qsiz];
            memset(FCS[i][j].Nedge, 0, sizeof(int) * qsiz);
            for (int kk = 0; kk < FCS[i][j].edges.size(); kk++)
            {
                FCS[i][j].Nedge[FCS[i][j].edges[kk].first]++;
            }
            uint32_t sd = 0;
            for (int kk = 0; kk < qsiz; kk++)
            {
                if (FCS[i][j].Nedge[kk] != 0)
                    sd++;
            }
            // No needed as to be here it passed the Degree Check
            if (de > sd)
                FCS[i][j].NedgeC = true;
        }
    }
}

/*Prune Node and edges, set node to deleted and edge to INVALID.
**Also find all neigboors of node and set the edges to the pruned node to INVALID
**Checks also if the remove edge makes the node to be pruned by the Nedge rule.
*/
inline void removeVertexAndEgjesFKNP(std::vector<std::vector<CSV>> &FCS, int i, int deli)
{
    VertexID vx1;
    uint32_t j;
    uint32_t k;
    for (j = 0; j < FCS[i][deli].edges.size(); j++)
    {
        // BSCHange
        if (FCS[i][deli].edges[j].first == INVALID)
            continue;
        vx1 = findIndBS(FCS, FCS[i][deli].edges[j].second, FCS[i][deli].edges[j].first);
        FCS[FCS[i][deli].edges[j].first][vx1].IPchange = true;
        FCS[FCS[i][deli].edges[j].first][vx1].change = true;
        // Mymutex.lock();
        FCS[FCS[i][deli].edges[j].first][vx1].Nedge[i]--;

        if (FCS[FCS[i][deli].edges[j].first][vx1].Nedge[i] == 0)
            FCS[FCS[i][deli].edges[j].first][vx1].NedgeC = true;
        // Mymutex.unlock();
        for (k = 0; k < FCS[FCS[i][deli].edges[j].first][vx1].edges.size(); k++)
        {
            if (FCS[FCS[i][deli].edges[j].first][vx1].edges[k].first == INVALID)
                continue;
            if (FCS[FCS[i][deli].edges[j].first][vx1].edges[k].first == i && FCS[FCS[i][deli].edges[j].first][vx1].edges[k].second == FCS[i][deli].ID)
            {
                FCS[FCS[i][deli].edges[j].first][vx1].edges[k].first = INVALID;
                break;
            }
        }
    }

    FCS[i][deli].edges.clear();
    FCS[i][deli].deleted = true;
}

inline bool OneHopEigenVG(CSV &cvertex, std::map<uint32_t, int> EvalNeigb, const Query& query, uint32_t *&flag, uint32_t *&updated_flag)
{
    uint32_t count2 = 0;
    uint32_t labela = 0;
    uint32_t k;
    uint32_t updated_flag_count = 0;
    for (k = 0; k < cvertex.edges.size(); k++)
    {
        if (cvertex.edges[k].first == INVALID)
            continue;
        labela = query.getVertexLabel(cvertex.edges[k].first);
        // check label only if it didnt pass the check yet
        if (EvalNeigb[labela] <= 0)
            continue;
        // Count only unique ID
        if (flag[cvertex.edges[k].second] != 1)
        {
            flag[cvertex.edges[k].second] = 1;
            updated_flag[updated_flag_count++] = cvertex.edges[k].second;
            EvalNeigb[labela]--;
            if (EvalNeigb[labela] == 0)
            {
                count2++;
                if (count2 == EvalNeigb.size())
                {
                    for (uint32_t aa = 0; aa <= (updated_flag_count + 1); aa++)
                    {
                        flag[updated_flag[aa]] = 0;
                    }
                    return true;
                }
            }
        }
    }
    for (uint32_t aa = 0; aa <= (updated_flag_count + 1); aa++)
    {
        flag[updated_flag[aa]] = 0;
    }
    return false;
}

/*Neighborhood NLF in candidate space.
 */

bool RefinementNV(std::vector<std::map<uint32_t, int>> NLabel, std::vector<std::vector<CSV>> &FCS, const Graph& data, const Query& query, uint32_t *&flag, uint32_t *&updated_flag)
{
    bool returnhere = true;
    uint32_t i;
    uint32_t j;
    uint32_t IDC;
    uint32_t pos;
    uint32_t NI;
    uint32_t query_vertex_num = query.getVertexCnt();
    uint32_t data_vertex_num = data.getVertexCnt();
    uint32_t query_graph_max_degree = query.getMaxDgree();
    uint32_t data_graph_max_degree = data.getMaxDgree();
    int qsiz = query.getVertexCnt();

    returnhere = false;
    int *flag1 = new int[data.getVertexCnt()];
    std::fill(flag1, flag1 + data.getVertexCnt(), -1);
    for (i = 0; i < qsiz; i++)
    {
        for (j = 0; j < FCS[i].size(); j++)
        {

            if (FCS[i][j].IPchange == false || FCS[i][j].deleted == true)
                continue;
            // 1 degree rule check

            if (!FCS[i][j].NedgeC)
                for (int ia = 0; ia < query.getVertexCnt(); ia++)
                {

                    if (FCS[i][j].Nedge[ia] == 1 && NLabel[i][query.getVertexLabel(ia)] > 1)
                    {
                        for (int aa = 0; aa < FCS[i][j].edges.size(); aa++)
                        {
                            if (ia == FCS[i][j].edges[aa].first)
                            {
                                IDC = FCS[i][j].edges[aa].second;
                                pos = aa;
                                break;
                            }
                        }
                        for (int aa = 0; aa < FCS[i][j].edges.size(); aa++)
                        {
                            if (FCS[i][j].edges[aa].second == IDC && aa != pos && FCS[i][j].edges[aa].first != INVALID)
                            {

                                uint32_t queryN = FCS[i][j].edges[aa].first;
                                NI = findIndBS(FCS, IDC, queryN);
                                for (int ao = 0; ao < FCS[queryN][NI].edges.size(); ao++)
                                {
                                    if (FCS[queryN][NI].edges[ao].first == i && FCS[queryN][NI].edges[ao].second == FCS[i][j].ID)
                                    {
                                        // remove neigboor
                                        FCS[queryN][NI].edges[ao].first = INVALID;
                                        FCS[queryN][NI].Nedge[i]--;
                                        if (FCS[queryN][NI].Nedge[i] == 0)
                                            FCS[queryN][NI].NedgeC = true;
                                        FCS[queryN][NI].change = true;
                                        FCS[queryN][NI].IPchange = true;

                                        // remove edge
                                        FCS[i][j].Nedge[queryN]--;
                                        if (FCS[i][j].Nedge[queryN] == 0)
                                            FCS[i][j].NedgeC = true;
                                        FCS[i][j].edges[aa].first = INVALID;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

            // Degree Check
            if (FCS[i][j].NedgeC)
            {
                removeVertexAndEgjesFKNP(FCS, i, j);
                returnhere = true;
            } // Neighborhood Check
            else if (!OneHopEigenVG(FCS[i][j], NLabel[i], query, flag, updated_flag))
            {
                removeVertexAndEgjesFKNP(FCS, i, j);
                returnhere = true;
            }
            else
            {
                FCS[i][j].IPchange = false;
            }
        }
    }

    delete[] flag1;
    return returnhere;
}

inline bool OHEPM(CSV &cvertex, uint32_t *&flag, uint32_t *&flagq, uint32_t *updated_flag, uint32_t *updated_flagq, int *IDDLC,
                  float **&LM, std::map<uint32_t, int> &EvalNeigb2, std::map<uint32_t, int> &EvalNeigb, const Query& query, std::vector<std::pair<VertexID, VertexID>> &q_curr, uint32_t Omax)
{

    // store the flag as the IDDLC+1
    // so when we look the position in array will be
    // x=flag[x]-1
    flag[cvertex.ID] = 1;         // value in LM matrix -1 same as count
    updated_flag[0] = cvertex.ID; //
    IDDLC[0]++;
    VertexID vx1 = 0;
    VertexID vx2 = 0;
    uint32_t count2 = 0;
    uint32_t labela = 0;
    // nedd IDDLC[1] reverse flag counter
    for (int k = 0; k < cvertex.edges.size(); k++)
    {
        if (cvertex.edges[k].first == INVALID)
            continue;
        flagq[cvertex.edges[k].first] = 1;
        if (flag[cvertex.edges[k].second] == 0)
        {
            flag[cvertex.edges[k].second] = IDDLC[0] + 1;
            updated_flag[IDDLC[0]] = cvertex.edges[k].second;
            IDDLC[0]++;
            if (IDDLC[1] > 0)
            {
                labela = query.getVertexLabel(cvertex.edges[k].first);
                EvalNeigb2[labela]--;
                EvalNeigb[labela]--;
                if (EvalNeigb2[labela] == 0)
                    IDDLC[1]--;
                if (EvalNeigb[labela] == 0)
                    IDDLC[2]--;
            }
            if ((IDDLC[0]) > Omax)
            {
                return true;
            }
            else
            {
                LM[0][IDDLC[0] - 1] = -1;
                LM[IDDLC[0] - 1][0] = -1;
            }
        }
        q_curr.emplace_back(cvertex.edges[k]);
    }
    return true;
}

inline bool SHEigen(std::vector<std::pair<VertexID, VertexID>> &q_curr, uint32_t *&flag, uint32_t *&flagq, uint32_t *&updated_flag, uint32_t *&updated_flagq, std::map<uint32_t, int> &EvalNeigb2,
                    int *IDDLC, std::vector<std::vector<CSV>> &FCS, float **&LM, const Query& query, uint32_t Omax, int qID, CSV &cvertex)
{
    using namespace std;

    pair<VertexID, VertexID> temp1;
    VertexID tempxx = 0;
    vector<VertexID> temp2;
    VertexID vx1 = 0;
    VertexID vx2 = 0;
    VertexID DN = 0;
    unordered_set<uint32_t> EdgeF;
    int kk = 0;
    uint32_t labela = 0;
    VertexID vtemp = 0;
    vector<uint32_t> tempx;
    uint32_t counter = 0;
    kk = 0;
    uint32_t tlqb = 0;
    while (kk < q_curr.size())
    {
        temp1 = q_curr[kk];
        kk++;

        if (temp1.first == INVALID)
        {
            continue;
        }

        tempxx = findIndBS(FCS, temp1.second, temp1.first);
        vx1 = flag[FCS[temp1.first][tempxx].ID] - 1;
        for (int i = 0; i < FCS[temp1.first][tempxx].edges.size(); i++)
        {
            tlqb = FCS[temp1.first][tempxx].edges[i].first;
            vtemp = FCS[temp1.first][tempxx].edges[i].second;
            if (tlqb == INVALID || tlqb == qID || vtemp == cvertex.ID)
                continue;
            flagq[tlqb] = 1;
            // with possible reverse flag but really  needed?

            if (flag[vtemp] == 0)
            {

                flag[vtemp] = (IDDLC[0] + 1);
                updated_flag[IDDLC[0]] = vtemp;
                vx2 = IDDLC[0];
                IDDLC[0]++;
                if (IDDLC[1] > 0)
                {
                    labela = query.getVertexLabel(tlqb);
                    EvalNeigb2[labela]--;
                    if (EvalNeigb2[labela] == 0)
                        IDDLC[1]--;
                }
                if (IDDLC[0] > Omax)
                {
                    return true;
                }
            }
            else
            {
                vx2 = flag[vtemp] - 1;
            }
            LM[vx1][vx2] = -1;
            LM[vx2][vx1] = -1;
        }
    }
    q_curr.clear();
    return (true);
}

inline bool SHEigenb(std::vector<std::pair<VertexID, VertexID>> &q_curr, uint32_t *&flag, uint32_t *&flagq, uint32_t *&updated_flag, uint32_t *&updated_flagq, std::map<uint32_t, int> &EvalNeigb2,
                     int *IDDLC, std::vector<std::vector<CSV>> &FCS, float **&LM, const Query& query, uint32_t Omax, int qID, CSV &cvertex, int beta)
{
    using namespace std;

    pair<VertexID, VertexID> temp1;
    VertexID tempxx = 0;
    vector<VertexID> temp2;
    VertexID vx1 = 0;
    VertexID vx2 = 0;
    VertexID DN = 0;
    unordered_set<uint32_t> EdgeF;
    int kk = 0;
    uint32_t labela = 0;
    VertexID vtemp = 0;
    vector<uint32_t> tempx;
    uint32_t counter = 0;
    kk = 0;
    uint32_t omaxUP = beta;
    while (kk < q_curr.size())
    {
        temp1 = q_curr[kk];
        kk++;

        if (temp1.first == INVALID)
        {
            continue;
        }

        tempxx = findIndBS(FCS, temp1.second, temp1.first);
        vx1 = flag[FCS[temp1.first][tempxx].ID] - 1;
        for (int i = 0; i < FCS[temp1.first][tempxx].edges.size(); i++)
        {
            counter++;
            if (FCS[temp1.first][tempxx].edges[i].first == INVALID || FCS[temp1.first][tempxx].edges[i].first == qID || FCS[temp1.first][tempxx].edges[i].second == cvertex.ID)
                continue;
            flagq[FCS[temp1.first][tempxx].edges[i].first] = 1;
            // with possible reverse flag but really  needed?
            vtemp = FCS[temp1.first][tempxx].edges[i].second;

            if (flag[vtemp] == 0)
            {
                if ((IDDLC[0]) > Omax)
                {
                    return true;
                }
                flag[vtemp] = IDDLC[0] + 1;
                updated_flag[IDDLC[0]] = vtemp;
                vx2 = IDDLC[0];
                IDDLC[0]++;
                if (IDDLC[1] > 0)
                {
                    labela = query.getVertexLabel(FCS[temp1.first][tempxx].edges[i].first);
                    EvalNeigb2[labela]--;
                    if (EvalNeigb2[labela] == 0)
                        IDDLC[1]--;
                }
            }
            else
            {
                vx2 = flag[vtemp] - 1;
            }
            LM[vx1][vx2] = -1;

            LM[vx2][vx1] = -1;

            if (counter > omaxUP)
            {
                IDDLC[0] = Omax + 1;
                return true;
            }
        }
    }
    q_curr.clear();
    return (true);
}

/*Prune Node and edges, set node to deleted and edge to INVALID.
**Also find all neigboors of node and set the edges to the pruned node to INVALID
**Checks also if the remove edge makes the node to be pruned by the Nedge rule.
*/

inline void removeVertexAndEgjesFK(std::vector<std::vector<CSV>> &FCS, int i, int deli)
{
    VertexID vx1;
    for (int j = 0; j < FCS[i][deli].edges.size(); j++)
    {
        if (FCS[i][deli].edges[j].first == INVALID)
            continue;
        vx1 = findIndBS(FCS, FCS[i][deli].edges[j].second, FCS[i][deli].edges[j].first);
        FCS[FCS[i][deli].edges[j].first][vx1].change = true;
        FCS[FCS[i][deli].edges[j].first][vx1].IPchange = true;
        FCS[FCS[i][deli].edges[j].first][vx1].Nedge[i]--;

        if (FCS[FCS[i][deli].edges[j].first][vx1].Nedge[i] == 0)
            FCS[FCS[i][deli].edges[j].first][vx1].NedgeC = true;
        for (int k = 0; k < FCS[FCS[i][deli].edges[j].first][vx1].edges.size(); k++)
        {
            if (FCS[FCS[i][deli].edges[j].first][vx1].edges[k].first == INVALID)
                continue;
            if (FCS[FCS[i][deli].edges[j].first][vx1].edges[k].first == i && FCS[FCS[i][deli].edges[j].first][vx1].edges[k].second == FCS[i][deli].ID)
            {
                FCS[FCS[i][deli].edges[j].first][vx1].edges[k].first = INVALID;
                break;
            }
        }
    }

    FCS[i][deli].edges.clear();
    FCS[i][deli].deleted = true;
}


/*Main function for Eigenvalue Pruning.
**Small Changes Added OMax and Omax2 First is limit for eigenvalues and second limit to twohop.
** We store the edges as triplets nd create a sparse Eigen MatrixXD
** We calculate eigenValues only if it passed the second pruning rule and the size of Matrix is less than oMax
*/
bool RFNV(std::vector<std::map<uint32_t, int>> NLabel, std::vector<std::map<uint32_t, int>> NLabel2, std::vector<std::vector<CSV>> &FCS, 
    const Query& query, float **&eigenVq1, std::vector<uint32_t> DM, int twohop, int alpha, int beta, uint32_t *&flag, uint32_t *&updated_flag, int **&VS){
    
    typedef Eigen::Triplet<double> T;
    using namespace std;
    using namespace Eigen;

    int MemSize=0;

    vector<T> tripletList;
    std::map<int, int> count_uniques;
    std::set<std::pair<int, int>> seen;
    std::vector<Triplet<double>> unique_triplets;

    int qsiz = query.getVertexCnt();


    uint32_t *flagq = new uint32_t[query.getVertexCnt()];
    std::fill(flagq, flagq + query.getVertexCnt(), 0);
    uint32_t *updated_flagq = new uint32_t[query.getVertexCnt()];
    std::fill(updated_flagq, updated_flagq + query.getVertexCnt(), 0);

    int IDDLC[3] = {0, 0, 0};
    bool returnhere = false;
    VertexID vertex = 0;
    vector<VertexID> temp2;
    vector<pair<VertexID, VertexID>> q_curr;
    int Eprun =30;
    if (Eprun>qsiz)
    Eprun=qsiz-1; 
    VertexID vertexDegree = 0;
    VertexID vertexlabel = 0;
    uint32_t SIDDSize = 0;
    bool continueE = false;
    bool con = true;
    uint32_t oMax;
    oMax = alpha;
    uint32_t oMax2;

    uint32_t *SIDN = new uint32_t[qsiz];
    uint32_t lb = query.getLabelCnt();
    float **LM = new float *[oMax + 1];
    for (int i = 0; i <= oMax; i++)
    {
        LM[i] = new float[oMax + 1];
        //this is wrong.
        memset(LM[i], 0, oMax + 1 * oMax + 1 * sizeof(float));
    }

    VectorXd evalues(Eprun);
    uint32_t i;
    uint32_t j;
    float sumD = 0;
    map<uint32_t, int> NLabelT;
    map<uint32_t, int> NLabelT1;

    for (int dd = 0; dd < qsiz; dd++)
    {
        i = dd;
        uint32_t NDL = query.getVertexDegree(i);

        for (j = 0; j < FCS[i].size(); j++)
        {
            if (FCS[i][j].deleted == true || FCS[i][j].change == false)
                continue;
            if (!FCS[i][j].NedgeC)
            {
                tripletList.clear();
                q_curr.clear();
                IDDLC[0] = 0;
                NLabelT = NLabel2[i];
                NLabelT1 = NLabel[i];
                IDDLC[1] = NLabel2[i].size();
                IDDLC[2] = NLabel[i].size();
                for (int aa = 0; aa < oMax; aa++)
                {
                    for (int bb = 0; bb < oMax; bb++)
                    {
                        LM[aa][bb] = 0;
                    }
                }
                for (int aa = 0; aa < qsiz; aa++)
                {
                    flagq[aa] = 0;
                }
                flagq[i] = 1;
                SIDDSize = 0;
                OHEPM(FCS[i][j], flag, flagq, updated_flag, updated_flagq, IDDLC, LM, NLabelT, NLabelT1,
                        query, q_curr, oMax);

                if (IDDLC[0] <= oMax)
                {

                    if (beta == 0)
                        SHEigen(q_curr, flag, flagq, updated_flag, updated_flagq, NLabelT, IDDLC, FCS, LM, query, oMax, i, FCS[i][j]);
                    else
                    {
                        SHEigenb(q_curr, flag, flagq, updated_flag, updated_flagq, NLabelT, IDDLC, FCS, LM, query, oMax, i, FCS[i][j], beta);
                    }
                    for (int aa = 0; aa < qsiz; aa++)
                        if (flagq[aa] == 1)
                            SIDDSize++;
                }
                if (IDDLC[0] <= oMax)
                    if ((IDDLC[0]) < DM[i] || IDDLC[1] > 0 || SIDDSize < DM[i])
                    { 
                        removeVertexAndEgjesFK(FCS, i, j);
                        returnhere = true;
                        for (int aa = 0; aa <= IDDLC[0]; aa++)
                        {
                            flag[updated_flag[aa]] = 0;
                        }
                        IDDLC[0] = oMax + 1;
                    }
                uint32_t s2 = IDDLC[0];
                uint32_t count2 = 0;

                if (s2 <= oMax && NDL != 1)

                {
                    int largest = 0;
                    con = true;
                    bool endFast = false;
                    uint32_t count1 = 0;
                    for (int l = 0; l < s2; l++)
                    {
                        count1 = 0;
                        for (int m = 0; m < s2; m++)
                        {
                            if (LM[l][m] == -1)
                            {
                                count1++;

                                tripletList.emplace_back(T(l, m, -1));
                            }
                        }
                        tripletList.emplace_back(T(l, l, count1));
                    }
                    int aa = 0;

                    if (tripletList.size() == s2 * s2)
                    {
                        evalues.resize(Eprun);

                        for (uint32_t ss = 0; ss < Eprun; ss++)
                        {
                            if (ss < s2)
                            {
                                evalues(ss) = s2;
                            }

                            else if (ss == IDDLC[0] - 1)
                                evalues(ss) = 0;
                            else
                                evalues(ss) = 0;
                        }
                    }
                    else
                    {
                        SparseMatrix<double> M(s2, s2);
                        M.setFromTriplets(tripletList.begin(), tripletList.end(), [](double a, double b)
                                            { return b; });
                        M.makeCompressed();

                        calcEigens1(M, Eprun, evalues, s2);
                    }
                    con = true;
                    sumD = 0;
                    //Eprun=4;
                    for (int dd = 0; dd < Eprun; dd++)
                    {
                        if (eigenVq1[i][dd] <= -1)
                            break;
                        if (evalues[dd] < eigenVq1[i][dd])
                        //if (evalues[dd] - eigenVq1[i][dd]<-0.0001)
                        {
                            if ((eigenVq1[i][dd] - evalues[dd]) > 0.0001)
                            {
                                con = false;
                                break;
                            }
                        }
                        // Eigen Ordering If we want to eigenvalues uncomment.
                        // Add the values up
                        // else
                        //     sumD += evalues[dd];
                    }
                    if (!con)
                    {

                        removeVertexAndEgjesFK(FCS, i, j);
                        returnhere = true;
                        for (int aa = 0; aa <= IDDLC[0]; aa++)
                        {
                            flag[updated_flag[aa]] = 0;
                        }
                    }

                    else
                    {
                        FCS[i][j].change = false;
                        // Eigen Ordering
                        //FCS[i][j].ED = sumD;
                        for (int aa = 0; aa <= IDDLC[0]; aa++)
                        {
                            flag[updated_flag[aa]] = 0;
                        }
                    }
                }
                else
                {
                    FCS[i][j].change = false;
                    for (int aa = 0; aa <= IDDLC[0]; aa++)
                    {
                        flag[updated_flag[aa]] = 0;
                    }
                }
            }
            else
            {
                removeVertexAndEgjesFK(FCS, i, j);
                returnhere = true;
            }
        }
    }

    //release memory
    for (int i = 0; i <= oMax; i++)
    {
        delete[] LM[i];
    }
    delete[] LM;
    delete[] SIDN;
    delete[] flagq;
    delete[] updated_flagq;

    return returnhere;
}

// ================ PILOS Main Function ===================
// Directly adopted from the original code
// ================ PILOS Main Function ===================

int PILOS(const Graph& data, const Query& query, float **&eigenVq1, int twohop, CandidateParam& canParam, float **&eigenVD1, int alpha, int beta)
{   
    using namespace std; 
    int totalCand = 0;
    int MemSize = 0;
    int qsiz = query.getVertexCnt();
    int dsiz = data.getVertexCnt();
    uint32_t **&candidates = canParam.candidates;
    uint32_t *&candidates_count = canParam.candidates_count;

    vector<vector<CSV>> FCS;
    FCS.reserve(qsiz);
    vector<uint32_t> DegreeK; // Discovered nodes for 2 hop
    vector<vector<pair<uint32_t, int>>> QueryNlabel;
    vector<map<uint32_t, int>> NLabel;  // Number of Labels 1hop
    vector<map<uint32_t, int>> NLabel2; // Number of Labels 2hop
    // Exctract 1hop label information for query graph


    uint32_t lb = query.getLabelCnt();
    ExtractNImap(query, NLabel);

    // Extract 2hop label information for query graph
    int **VS = NULL;
    VS = new int *[qsiz];
    for (int aa = 0; aa < qsiz; aa++)
    {
        VS[aa] = new int[qsiz];
        for (int bb = 0; bb < qsiz; bb++)
        {
            VS[aa][bb] = -1;
        }
    }
    ExtractUI2h(query,DegreeK, NLabel2, VS);


    Vertices(FCS, data, query, eigenVq1, NLabel, eigenVD1);
    
    totalCand = 0;
    int count = 0;
    int Tcount = 0;

    // Add Edges between nodes in candidate space
    uint32_t *flag = new uint32_t[data.getVertexCnt()+5];
    std::fill(flag, flag + data.getVertexCnt()+5, 0);
    uint32_t *updated_flag = new uint32_t[data.getVertexCnt()+5];
    std::fill(updated_flag, updated_flag + data.getVertexCnt()+5, 0);
    
    EdgesCSBasicRL(FCS, data, query, flag, updated_flag);
    // Get candidate nodes neigborhood information for fast pruning¨
    // Initial Pruning on Candidate Space


    while (InitPrunTCSR(FCS, query))
        clearWrong(FCS);

    fillEN(FCS, query);

    // Neigborhood Pruning
    for (int i = 0; i < qsiz; i++)
    {
        for (int j = 0; j < FCS[i].size(); j++)
        {
            std::sort(FCS[i][j].edges.begin(), FCS[i][j].edges.end(), [](const auto &a, const auto &b)
                      { return a.first < b.first; });
        }
    }
    int cc = 0;
    while (RefinementNV(NLabel, FCS,data, query, flag, updated_flag)){
        clearWrong(FCS);
    }
    uint32_t mc = 3;
    while (RFNV(NLabel, NLabel2, FCS, query, eigenVq1, DegreeK, twohop, alpha, beta, flag, updated_flag, VS) && mc < 5)
    {
        mc++;
        clearWrong(FCS);
        while (RefinementNV(NLabel, FCS, data, query, flag, updated_flag))
        {
            clearWrong(FCS);
        }
    }
    delete[] flag;
    delete[] updated_flag;
    clearWrong(FCS);
    // allocateBufferFCS1(FCS, query, candidates, candidates_count, EWeight);
    // release VS

    // Release memory
    for (int i = 0; i < qsiz; i++)
    {
        for (int j = 0; j < FCS[i].size(); j++)
        {
            delete[] FCS[i][j].Nedge;
        }
    }
    for (int aa = 0; aa < qsiz; aa++)
    {
        delete[] VS[aa];
    }
    delete[] VS;


    for (int i = 0; i < qsiz; i++)
    {   
        for (int j = 0; j < FCS[i].size(); j++)
        {
            candidates[i][j] = FCS[i][j].ID;
        }
        candidates_count[i] = FCS[i].size();
        totalCand = candidates_count[i] + totalCand;
        if(candidates_count[i] == 0)
        {
            printf("Pilos filter invalid, exit\n");
            return false; // No candidates found for this query vertex
        }
    }

    return true;
}



bool _Pilos_Filter(const Graph& data, const Query& query, int twohop, CandidateParam& canParam, float **&eigenVD1, int alpha, int beta)
{
    uint32_t **&candidates = canParam.candidates;
    uint32_t *&candidates_count = canParam.candidates_count;
    int sizq = query.getVertexCnt();
    uint32_t Eprun = sizq - 3;
    Eprun = 30;
    Eigen::MatrixXd eigenVq1(sizq, Eprun);
    int oMax = sizq * 3;
    oMax = 10000;
    MTcalc12(query, query.getMaxDgree(), eigenVq1, true, Eprun, oMax);
    float **eigenQ = NULL;
    eigenQ = new float *[sizq];
    for (uint32_t i = 0; i < sizq; ++i)
    {
        eigenQ[i] = new float[Eprun];
        for (uint32_t j = 0; j < Eprun; j++)
        {
            eigenQ[i][j] = eigenVq1(i, j);
        }
    }
    
    bool ret = PILOS(data, query, eigenQ, twohop, canParam, eigenVD1, alpha, beta);
    // Release memory
    for (uint32_t i = 0; i < sizq; ++i)
    {
        delete[] eigenQ[i];
    }
    delete[] eigenQ;

    return ret;
}