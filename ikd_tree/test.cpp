#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>

#include <pcl/point_types.h>

#define FINF 3.0e10
#define EPS 1e-5

using namespace std::chrono_literals;
typedef std::vector<std::vector<float>> Range;


template<class PointType>
struct PCLCmp {
    PCLCmp(int type = 0) :
    type(type) { };
    bool operator()(PointType &a, PointType &b) {
        switch (type) {
            case 0:
                return a.x < b.x;
            case 1:
                return a.y < b.y;
            case 2:
                return a.z < b.z;
            default:
                return a.x < b.x;
        }
    }
private:
    int type;
};

struct RangeCmp {
    bool operator()(Range &a, Range &b) {
        return (std::abs(a[0][0] - b[0][0]) < EPS && std::abs(a[0][1] - b[0][1]) < EPS &&
                std::abs(a[1][0] - b[1][0]) < EPS && std::abs(a[1][1] - b[1][1]) < EPS &&
                std::abs(a[2][0] - b[2][0]) < EPS && std::abs(a[2][1] - b[2][1]) < EPS);
    }
};

struct RangeHash {
    uint64_t operator()(Range &a) {
        return static_cast<uint64_t>(1e6 * (a[0][0] * 17 + a[1][0] * 5 + a[2][0] * 2 + 17));
    }
};

// template<class Cmp, class Hash>
class HashQueue {
public:
    HashQueue() = default;
    HashQueue( RangeCmp cmp, RangeHash hash, int expected_element=1000, float bf=0.75) :
    cmp(cmp),
    table_size(static_cast<int>(expected_element / bf)),
    bf(bf) {
        DataField default_val;
        table = std::vector<DataField>(table_size, default_val);
    }

    bool push(Range &new_data) {
        uint64_t hash_val = hash(new_data);
        hash_val %= table_size;
        while (true) {
            if (table[hash_val].occupied == false) {
                table[hash_val].data = new_data;
                table[hash_val].occupied = true;
                queue.push(new_data);
                return true;
            }
            else {
                if (cmp(table[hash_val].data, new_data)) {
                    return false;
                }
                else {
                    hash_val = (hash_val + 1) % table_size;
                }
            }
        }
        if (static_cast<float>(queue.size()) / table_size > bf) {
            throw std::runtime_error("Hash Overflow!");
        }
    }

    Range pop(void) {
        if (queue.empty()) {
            throw std::runtime_error("Requests already empty!");
        }
        Range retval = queue.front();
        uint64_t hash_val = hash(retval) % table_size;
        while (true) {
            if (cmp(table[hash_val].data, retval)) {
                table[hash_val].occupied = false;
                break;
            }
            else {
                hash_val = (hash_val + 1) % table_size;
                conflict_cnt ++;
            }
        }
        queue.pop();
        return retval;
    }

    bool is_empty() {
        return queue.empty();
    }

    int size() {
        return queue.size();
    }

    float get_usage() {
        return static_cast<float>(queue.size()) / table_size;
    }

    int get_conflict_cnt() {
        return conflict_cnt;
    }

private:
    std::queue<Range> queue;
    struct DataField {
        Range data = std::vector<std::vector<float>>(3, std::vector<float>(2, 0));
        bool occupied = false;
    };
    std::vector<DataField> table;
    int conflict_cnt = 0;
    uint64_t table_size;
    RangeCmp cmp;
    RangeHash hash;
    float bf;
};

template <typename T>
class NormalQueue {
public:
    NormalQueue() = default;
    void push(T &data) {
        {
            queue.push(data);
        }
        q_size ++;
    }

    T pop(void) {
        T retval;
        {
            retval = queue.front();
            queue.pop();
        }
        q_size -= 1;
        return retval;
    }

    int size() {
        return q_size;
    }

    bool is_empty() {
        return q_size <= 0;
    }

private:
    std::queue<T> queue;
    // std::mutex mtx;
    int q_size = 0;
};

// Cmp() implement <, get bigger root heap
template <typename T, typename Cmp>
class PriorityQueue {
public:

    PriorityQueue(int heap_size, Cmp cmp) :
    data(std::vector<T>(heap_size)),
    cmp(cmp),
    heap_size(heap_size),
    counter(0) { }

    T& top() {
        if (counter <= 0) { throw std::runtime_error("Zero size heap visited!"); }
        return data[0];
    }

    bool is_empty(void) {
        return counter <= 0;
    }

    void clear() {
        counter = 0;
    }

    int size() {
        return counter;
    }

    void push(T &element) {
        if (counter >= heap_size) {
            throw std::runtime_error("Max heap size reached, set bigger");
        }
        int idx = counter;
        while (idx > 0) {
            int father = (idx - 1) / 2;
            if (cmp(data[father], element)) {
                // drift up bigger one.
                data[idx] = data[father];
                idx = father;
            }
            else { break; }
        }
        data[idx] = element;
        counter ++;
    }

    void pop(void) {
        if (counter <= 0) { return; }
        counter --;
        T tmp_data = data[counter];
        int left = 1;
        while (left < counter) {
            if (left + 1 < counter && cmp(data[left], data[left + 1])) {
                left ++;
            }
            if (cmp(tmp_data, data[left])) {
                data[static_cast<int>((left - 1) / 2)] = data[left];
                left = left * 2 + 1;
            }
            else { break; }
        }
        data[static_cast<int>((left - 1) / 2)] = tmp_data;
    }

private:
    Cmp cmp;
    std::vector<T> data;
    int counter = 0;
    int heap_size;
};

template <typename PointType>
class IKDTree {
public:
    struct Node;
    struct RebuildBuffer;
    struct OpContext; 
    enum class operation { pnt_add, pnt_delete, box_delete, push_down, delete_tree };

    struct Node {
        enum class status { none, ins, del, ser };
        Node* father = nullptr, *left_son = nullptr, *right_son = nullptr;
        float x = 0, y = 0, z = 0;
        uint8_t dim = 0;
        // pull up /pushdown properties
        Range valid_range = std::vector<std::vector<float>>(3, std::vector<float>(2, 0)); 
        bool need_push_down = false;
        bool deleted = false;
        bool tree_deleted = false;

        float del_balance = 0.3, height_balance = 0.5;
        bool balanced = true;
        int tree_total_size = 1;
        int tree_valid_size = 1;

        // rebuild related
        status st = status::none;
        std::mutex mtx;
        RebuildBuffer* rebuild_buffer_ptr = nullptr;
        bool working_flag = false;
        // int search_counter = 0;
        
        static void copy(Node *dst, Node *src) {
            dst->father = src->father;
            dst->left_son = src->left_son;
            dst->right_son = src->right_son;

            dst->x = src->x; dst->y = src->y; dst->z = src->z;

            dst->dim = src->dim;
            dst->valid_range = src->valid_range;
            dst->need_push_down = src->need_push_down;
            dst->deleted = src->deleted;
            dst->tree_deleted = src->tree_deleted;

            dst->del_balance = src->del_balance;
            dst->balanced = src->balanced;
            dst->tree_total_size = src->tree_total_size;
            dst->tree_valid_size = src->tree_valid_size;
            // Other properties are held.
        }
    };
    struct RebuildBuffer {
        enum class status { logging, finish};
        // All the members not need mtx, except safe queue
        Node *node = nullptr;  // th1 w, th2 r
        bool valid = true;  // th2 r
        int tree_size = 0;  // th1 w
        status st = status::logging;
        NormalQueue<OpContext> logger;
    };
    struct RebuildBufferCmp {
        bool operator()(RebuildBuffer *a, RebuildBuffer *b) {
            return a->tree_size < b->tree_size;
        }  
    };

    struct OpContext {
        operation op;
        PointType pnt;
        Range range;
        bool enable_down;
    };

    struct PointCmp {
        float dist_sq = 0;
        PointType point;
        bool operator()(PointCmp &a, PointCmp &b) {
            return a.dist_sq < b.dist_sq;
        }
    };

    IKDTree(float del_criterion=0.5, float height_criterion=0.8, float voxel_size=0.2) :
    cmpx(PCLCmp<PointType>(0)),
    cmpy(PCLCmp<PointType>(1)),
    cmpz(PCLCmp<PointType>(2)),
    rebuild_queue(PriorityQueue<RebuildBuffer*, RebuildBufferCmp>(1000, cmp_rebuild)),
    del_criterion(del_criterion),
    height_criterion(height_criterion),
    voxel_size(voxel_size) {
        static_root = new Node;
        init_dummy_node();
        if (height_criterion <= 0.5) { std::cerr << "WARNING, height criterion should larger than 0.5";}

        int dsp_size = std::pow(static_cast<int>(1 / voxel_size), 2) * 20000 + 17;
        downsampling_register = HashQueue(cmp_down, hash_down, dsp_size);

        second_th = std::thread(&IKDTree::guard_thread, this);
    }

    ~IKDTree(void) {
        terminate = true;
        second_th.join();
        free_tree(static_root);
    }

    void insert(std::vector<PointType> &point_storage, bool enable_downsampling, bool enable_rebuilding) {
        if (static_root->left_son == nullptr) {
            static_root->left_son = build_tree(point_storage, 0,
                                               point_storage.size(),
                                               static_root, enable_downsampling);
            root_ptr = static_root->left_son;
        }
        else {
            for (auto &i : point_storage) {
                point_insert(root_ptr, i, enable_downsampling, enable_rebuilding);
            }
        }
    }

    // build [begin, end)
    // No thread safe required
    Node*
    build_tree(std::vector<PointType> &point_storage, int begin, int end, Node* father, bool enable_downsampling) {
        // Return condition
        if (begin >= end) { return nullptr; }
        // Axis selection
        std::vector<std::vector<float>> range_min_max = {{FINF, -FINF}, {FINF, -FINF}, {FINF, -FINF}};

        for (int idx = begin; idx < end; idx ++) {
            range_min_max = merge_range(point_storage[idx], range_min_max);
        }

        float range[3] = {range_min_max[0][1] - range_min_max[0][0],
            range_min_max[1][1] - range_min_max[1][0],
            range_min_max[2][1] - range_min_max[2][0]};
        uint8_t idx = 2;
        if (range[idx] < range[1]) { idx = 1; }
        if (range[idx] < range[0]) { idx = 0; }

        int nth = static_cast<int>(begin + (end - begin) / 2);
        auto it_beg = point_storage.begin() + begin;
        auto it_end = point_storage.begin() + end;
        auto it_nth = point_storage.begin() + nth;
        switch (idx) {
            case 0:
                std::nth_element(it_beg, it_nth, it_end, cmpx);
                break;
            case 1:
                std::nth_element(it_beg, it_nth, it_end, cmpy);
                break;
            case 2:
                std::nth_element(it_beg, it_nth, it_end, cmpz);
                break;
            default:
                throw std::runtime_error("Why this condition happens?");
        }

        Node* node = pnt_2_node(point_storage[nth]);
        node->dim = idx;
        node->father = father;
        node->right_son = build_tree(point_storage, begin, nth, node, enable_downsampling);
        node->left_son = build_tree(point_storage, nth + 1, end, node, enable_downsampling);
        update(node);
        if (enable_downsampling) { regist_downsampling_request(point_storage[nth]); }
        return node;
    }

    // Thread safe
    void point_insert(Node *node, PointType &point, bool enable_downsampling, bool enable_rebuilding) {
        if (node == nullptr) {
            throw std::runtime_error("don't insert point to nullptr!");
        }
        
        // Rebuilding Node
        std::unique_lock<std::mutex> node_lock(node->mtx); 
        if (node->rebuild_buffer_ptr != nullptr) {
            // background building
            if (node->rebuild_buffer_ptr->st == RebuildBuffer::status::logging) {
                OpContext op;
                op.op = operation::pnt_add;
                op.pnt = point;
                op.enable_down = enable_downsampling;
                node->rebuild_buffer_ptr->logger.push(op);
                node_lock.unlock();
                return;
            }
            else {
                // Replacing root 
                auto start_time = std::chrono::system_clock::now();
                while(true) {
                    node_lock.unlock(); 

                    std::this_thread::sleep_for(100ns);
                    auto time_dur = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now() - start_time
                    );
                    if (time_dur > 500us) {
                        throw std::runtime_error("Waiting over 500us at insert.");
                    }

                    node_lock.lock();
                    if (node->working_flag == false) {
                        break;
                    }
                }
            }
        }
        node->working_flag = true;
        node->st = Node::status::ins;
        node_lock.unlock();

        propagate(node);

        float cmp_node, cmp_point;
        switch (node->dim) {
            case 0:
                cmp_node = node->x;
                cmp_point = point.x;
                break;
            case 1:
                cmp_node = node->y;
                cmp_point = point.y;
                break;
            case 2:
                cmp_node = node->z;
                cmp_point = point.z;
                break;
            default:
                throw std::runtime_error("no this dim in point insert.");
        };
        // Equal
        if (dist_sq(node, point) < EPS) {
            node->deleted = false;
        }
        else if (cmp_point < cmp_node) {
            if (node->right_son == nullptr) {
                Node *new_node = pnt_2_node(point);
                new_node->father = node;
                node->right_son = new_node;
                new_node->dim = (node->dim + 1) % 3;
            }
            else {
                // Once insert one point, register it once.
                point_insert(node->right_son, point, false, enable_rebuilding);
            }
        }
        else {
            if (node->left_son == nullptr) {
                Node *new_node = pnt_2_node(point);
                new_node->father = node;
                node->left_son = new_node;
                new_node->dim = (node->dim + 1) % 3;
            }
            else {
                point_insert(node->left_son, point, false, enable_rebuilding);
            }
        }
        update(node);
        {
            std::lock_guard<std::mutex> lock(node->mtx);
            node->working_flag = false;
            node->st = Node::status::none;
        }

        if (enable_downsampling) { regist_downsampling_request(point); }
        if (enable_rebuilding && (!node->balanced)) {
            rebuild_subtree(node);
        }
    }

    void point_delete(Node *node, PointType &point, bool enable_rebuild) {
        if (node == nullptr || node->tree_deleted) { return; }

        std::unique_lock<std::mutex> lock(node->mtx); 
        if (node->rebuild_buffer_ptr != nullptr) {
            if (node->rebuild_buffer_ptr->st == RebuildBuffer::status::logging) {
                OpContext op;
                op.op = operation::pnt_delete;
                op.pnt = point;
                node->rebuild_buffer_ptr->logger.push(op);
                lock.unlock();  // Note this!!!
                return;
            }
            else {
                // Replacing root 
                auto start_time = std::chrono::system_clock::now();
                while(node->working_flag == true) {
                    lock.unlock(); 

                    std::this_thread::sleep_for(100us);
                    auto time_dur = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now() - start_time
                    );
                    if (time_dur > 500us) {
                        throw std::runtime_error("Waiting over 500us at insert.");
                    }

                    lock.lock();
                }
            }
        }
        
        node->working_flag = true;
        node->st = Node::status::del;
        lock.unlock();

        propagate(node);

        if (dist_node_pnt(node, point) <= EPS) {
            if (!node->deleted) {
                 node->deleted = true;
                 // update(node);
            }
        }
        else {
            float cmp_node, cmp_point;
            switch (node->dim) {
                case 0:
                    cmp_node = node->x;
                    cmp_point = point.x;
                    break;
                case 1:
                    cmp_node = node->y;
                    cmp_point = point.y;
                    break;
                case 2:
                    cmp_node = node->z;
                    cmp_point = point.z;
                    break;
                default:
                    throw std::runtime_error("no this dim in point insert.");
            }
            if (cmp_point < cmp_node) {
                if (node->right_son != nullptr) {
                    point_delete(node->right_son, point, enable_rebuild);
                    // update(node);
                }
            }
            else {
                if (node->left_son != nullptr) {
                    point_delete(node->left_son, point, enable_rebuild);
                    // update(node);
                }
            }
        }
        update(node);

        {
            std::lock_guard<std::mutex> lock(node->mtx);
            node->working_flag = false;
            node->st = Node::status::none;
        }
 
        if (enable_rebuild && (!node->balanced)) {
            rebuild_subtree(node);
        }
    }

    void box_delete(Node *node, Range &box, bool enable_rebuild) {
        if (node == nullptr || node->tree_deleted) { return; }

        std::unique_lock<std::mutex> lock(node->mtx);
        if (node->rebuild_buffer_ptr != nullptr) {
            if (node->rebuild_buffer_ptr->st == RebuildBuffer::status::logging) {
                OpContext op;
                op.op = operation::box_delete;
                op.range = box;
                node->rebuild_buffer_ptr->logger.push(op);                
                lock.unlock();
                return;
            }
            else {
                auto start_time = std::chrono::system_clock::now();
                while(node->working_flag == true) {
                    lock.unlock(); 
                    std::this_thread::sleep_for(100us);
                    auto time_dur = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now() - start_time
                    );
                    if (time_dur > 500us) {
                        throw std::runtime_error("Waiting over 500us at insert.");
                    } 
                    lock.lock();
                }
            }
        }
        
        node->working_flag = true;
        node->st = Node::status::del;
        lock.unlock();

        propagate(node);

        if (is_subset(node->valid_range, box)) {
            set_tree_del(node);        
            lock.lock();
            node->working_flag = false;
            node->st = Node::status::none;
            lock.unlock();
            return;
        }
        else if (is_intersect(node->valid_range, box)) {
            if (is_subset(node, box)) {
                node->deleted = true;
            }
            if (node->left_son != nullptr) {
                box_delete(node->left_son, box, enable_rebuild);
            }
            if (node->right_son != nullptr) {
                box_delete(node->right_son, box, enable_rebuild);
            }        
            update(node);

            lock.lock();
            node->working_flag = false;
            node->st = Node::status::none;
            lock.unlock();

            if (enable_rebuild && (!node->balanced)) {
                rebuild_subtree(node);
            }
        }
        else {
            lock.lock();
            node->working_flag = false;
            node->st = Node::status::none;
            lock.unlock();
            return;
        }

    }

    void knn_search(Node *node, PointType &target, int k, float max_range, std::vector<float> &dist_sq, std::vector<PointType> &points) {
        PointCmp cmp;
        PriorityQueue<PointCmp, PointCmp> q(k + 1, cmp);
        _knn_search(root_ptr, q, k, target, max_range * max_range);

        points.clear();
        dist_sq.clear();
        for (int i = 0; i < k; i++) {
            if (q.size() <= 0) {
                break;
            }
            points.push_back(q.top().point);
            dist_sq.push_back(q.top().dist_sq);
            q.pop();
        }
        // if (points.size() < k) {
        //     std::cout << "WARNING: Not found enough points, find " << points.size() << " expected " << k << std::endl;
        // }
    }

    void _knn_search(Node *node, PriorityQueue<PointCmp, PointCmp> &heap, int k, PointType &target, float max_range_sq) {
        if (node == nullptr || node->tree_deleted) { return; }

        std::unique_lock<std::mutex> lock(node->mtx);
        if (node->rebuild_buffer_ptr != nullptr) {
            auto start_time = std::chrono::system_clock::now();
            while (node->working_flag != false) {
                // Changing root, don't search now.
                lock.unlock();
                auto dur = std::chrono::duration_cast<std::chrono::microseconds> (
                    std::chrono::system_clock::now() - start_time);
                if (dur > 5ms) {
                    throw std::runtime_error("Waiting over 500us at search.");
                }
                std::this_thread::sleep_for(80us);                
                lock.lock(); 
            }
        }

        node->working_flag = true;
        node->st = Node::status::ser;
        lock.unlock();

        propagate(node);

        float possible_shortest_sq = min_box_dist_sq(node->valid_range, target);
        if (possible_shortest_sq > max_range_sq || (heap.size() >= k && possible_shortest_sq > heap.top().dist_sq)) {
            lock.lock();
            node->working_flag = false;
            lock.unlock();
            return;
        }

        // add point when valid and smaller than
        float cur_dist_sq = dist_sq(node, target);
        if (cur_dist_sq < max_range_sq && !node->deleted && (heap.size() < k || (!heap.is_empty() && cur_dist_sq < heap.top().dist_sq))) {
            if (heap.size() >= k) {
                heap.pop();
            }
            PointCmp new_point;
            new_point.dist_sq = cur_dist_sq;
            new_point.point = node_2_pnt(node);
            heap.push(new_point);
        }

        float possible_short_sq_left = (node->left_son != nullptr)?
                                        min_box_dist_sq(node->left_son->valid_range, target):
                                        FINF;
        float possible_short_sq_right = (node->right_son != nullptr)?
                                        min_box_dist_sq(node->right_son->valid_range, target):
                                        FINF;

        if (possible_short_sq_left <= possible_short_sq_right) {
            // if no size and no possibility.
            if (node->left_son != nullptr &&
                (heap.size() < k ||
                possible_short_sq_left < heap.top().dist_sq )) {
                _knn_search(node->left_son, heap, k, target, max_range_sq);
            }

            if (node->right_son != nullptr &&
                (heap.size() < k ||
                possible_short_sq_right < heap.top().dist_sq)) {
                _knn_search(node->right_son, heap, k, target, max_range_sq);
            }
        }
        else {
            if (node->right_son != nullptr &&
                (heap.size() < k ||
                possible_short_sq_right < heap.top().dist_sq)) {
                _knn_search(node->right_son, heap, k, target, max_range_sq);
            }
            if (node->left_son != nullptr &&
                (heap.size() < k ||
                possible_short_sq_left < heap.top().dist_sq)) {
                _knn_search(node->left_son, heap, k, target, max_range_sq);
            }
        }
        lock.lock();
        node->working_flag = false;
        node->st = Node::status::none; 
        lock.unlock();
    }

    void regist_downsampling_request(PointType &point) {
        Range down_request (3, std::vector<float>(2, 0));
        float xyz[3];
        xyz[0] = point.x;
        xyz[1] = point.y;
        xyz[2] = point.z;
        for (int i = 0; i < 3; i++) {
            down_request[i][0] = voxel_size * std::floor(xyz[i] / voxel_size);
            down_request[i][1] = down_request[i][0] + voxel_size;
        }

        downsampling_register.push(down_request);
        downsampling_requests ++;
    }

    // Downsampling. We execute rebuild altogether.
    void downsampling_execute(void) {
        Range down_range;
        PointType target;
        std::vector<PointType> pnt_buf(1);
        std::vector<float> dist(1);
        float half_voxel = voxel_size / 2;
        while (!downsampling_register.is_empty()) {
            down_range = downsampling_register.pop();
            target.x = down_range[0][0] + half_voxel;
            target.y = down_range[1][0] + half_voxel;
            target.z = down_range[2][0] + half_voxel;
            knn_search(root_ptr, target, 1, voxel_size * 1.2, dist, pnt_buf);
            if (pnt_buf.size() > 0) {
                // printf("%f\n", dist[0]);
                box_delete(root_ptr, down_range, true);
                point_insert(root_ptr, pnt_buf[0], false, true);
            }
            downsampling_executes ++;
            // std::cout << downsampling_register.size() << std::endl;
        }
    }

    void guard_thread(void) {
        std::vector<PointType> flatten_buffer;
        std::unique_lock<std::mutex> rebuild_queue_lock(rebuild_q_mtx, std::defer_lock);
        OpContext op_buf;

        RebuildBuffer *request;
        bool empty_tree = false;
        while (!terminate) {    
            // 1. get request (validate it)
            rebuild_queue_lock.lock();
            if (!rebuild_queue.is_empty()) {
                request = rebuild_queue.top();
                rebuild_queue.pop();
                rebuild_queue_lock.unlock();

                if (request->valid == false) {
                    continue;
                }
            }
            else {
                rebuild_queue_lock.unlock();
                std::this_thread::sleep_for(20us);
                continue;
            }
            
            // Already get request
            // 2. cpy tree, lock this node when cpy.
            flatten_buffer.clear();
            {
                auto start = std::chrono::system_clock::now();
                std::unique_lock<std::mutex> cpy_lock(request->node->mtx);
                while (request->node->working_flag) {
                    cpy_lock.unlock();
                    std::this_thread::sleep_for(20us);
                    if (std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now() - start
                    ) > 100ms) {
                        throw std::runtime_error("Waiting too long in second thread flatten");
                    }
                    cpy_lock.lock(); 
                }
                request->node->working_flag = true;
                cpy_lock.unlock();
                
                if (request->node->tree_valid_size == 0) {
                    // Keep at least one.
                    PointType least = node_2_pnt(request->node);
                    flatten_buffer.push_back(least);
                    empty_tree = true;
                }
                flatten(request->node, flatten_buffer);
                cpy_lock.lock();
                request->node->working_flag = false;
                cpy_lock.unlock();
            }

          
            Node *tmp_root = build_tree(flatten_buffer, 0, flatten_buffer.size(), request->node->father, false);
            if (empty_tree) {
                set_tree_del(tmp_root);
                empty_tree = false;
            }

            // 3. execute and finish
            std::unique_lock<std::mutex> node_lock(request->node->mtx, std::defer_lock);            

            while (true) {
                // If there are pending operations...
                node_lock.lock();
                if (!request->logger.is_empty()) {
                    op_buf = request->logger.pop();
                    node_lock.unlock();
                    switch (op_buf.op) {
                        case operation::pnt_add:
                            point_insert(tmp_root, op_buf.pnt, op_buf.enable_down, false);
                            break;
                        case operation::pnt_delete:
                            point_delete(tmp_root, op_buf.pnt, false);
                            break;
                        case operation::box_delete:
                            box_delete(tmp_root, op_buf.range, false);
                            break;
                        case operation::push_down:
                            set_tree_del(tmp_root);
                            break;
                        default:
                            throw std::runtime_error("Hum? A new operation?");
                            break;
                    }
                }
                else {  
                    if (request->node->working_flag == true) {
                        // working, ins, del, searching...
                        node_lock.unlock();
                        std::this_thread::sleep_for(20us);
                    }
                    else {
                        // end. And no pending operation.
                        request->st = RebuildBuffer::status::finish;
                        break;
                    }
                }
            }
            // The node is locked now.
            // Main thread waiting.
            // Cpy tree and clear flags.
            Node::copy(request->node, tmp_root);
            request->node->rebuild_buffer_ptr = nullptr;
            // delete request->node->rebuild_buffer_ptr;
            request->node->working_flag = false; 
            Node *father = request->node->father;
            node_lock.unlock();

            // update father after rebuilding.
            while (father != nullptr && father != static_root) {
                {
                    std::lock_guard<std::mutex> father_lock(father->mtx);
                    if ((father->working_flag == true && father->st != Node::status::ser) ||
                         father->need_push_down) {
                        break;
                    }
                    update(father);
                }
                father = father->father;
            }
            // std::cout << "call second and change root!" << std::endl;
        }
    }

    void rebuild_subtree(Node *root) {
        if (root->tree_total_size <= 100) {
            bool empty_tree = (root->tree_valid_size == 0);
            rebuild_buffer_main.clear();
            flatten(root, rebuild_buffer_main);
            if (empty_tree) {
                PointType least;
                least.x = root->x;
                least.y = root->y;
                least.z = root->z;
                rebuild_buffer_main.push_back(least);                
            }
            Node * new_root = build_tree(rebuild_buffer_main, 0, 
                rebuild_buffer_main.size(), root->father, false);
            Node *tmp_left = root->left_son;
            Node *tmp_right = root->right_son;
            if (empty_tree) {
                set_tree_del(new_root);
            }
            Node::copy(root, new_root);
            free_tree(tmp_left);
            free_tree(tmp_right);
            // std::cout << "rebuild in main thread" << std::endl;
        }
        else {
            root->rebuild_buffer_ptr = new RebuildBuffer;
            root->rebuild_buffer_ptr->node = root;
            root->rebuild_buffer_ptr->tree_size = root->tree_total_size;
            root->rebuild_buffer_ptr->valid = true;
            std::lock_guard<std::mutex> lock(rebuild_q_mtx);
            rebuild_queue.push(root->rebuild_buffer_ptr);
            // std::cout << "register rebuild task" << std::endl;
        }
    }


    void flatten(Node *node, std::vector<PointType> &storage) {
        if (node == nullptr) { return; }
        propagate(node);
        if (!node->deleted) {
            storage.push_back(node_2_pnt(node));
        }
        if (node->left_son != nullptr) { flatten(node->left_son, storage); }
        if (node->right_son != nullptr) { flatten(node->right_son, storage); }
    }

    void free_tree(Node *node) {
        if (node == nullptr) { return; }
        free_tree(node->left_son);
        free_tree(node->right_son);
        // excluded, safe.
        if (node->rebuild_buffer_ptr) {
            node->rebuild_buffer_ptr->valid = false;
        }
        delete node;
    }

    // Act after any status change (delete, tree delete, need_pushdown)
    // Make sure every node you visit is updated after exit
    // WARNING: RACE HERE. If visit replacing son node, racing occurs.
    // There are no need for current node to be locked. 
    void update(Node* node) {
        if (node == nullptr) { return; }

        if (node->need_push_down) { throw std::runtime_error("Never need push down when update"); }
        Node *left_son;
        Node *right_son;
        left_son = (node->left_son == nullptr)? dummy_node_left: node->left_son;
        right_son = (node->right_son == nullptr)? dummy_node_right: node->right_son;

        std::unique_lock<std::mutex> left_lock(left_son->mtx);
        if (left_son->rebuild_buffer_ptr != nullptr) {
            if (left_son->working_flag != false) {
                auto start = std::chrono::system_clock::now();
                while (left_son->working_flag != false) {
                    left_lock.unlock();
                    auto dur = std::chrono::duration_cast<std::chrono::microseconds> (
                        std::chrono::system_clock::now() - start
                    );
                    if (dur > 1ms) {
                        throw std::runtime_error("Left son over when update");
                    }
                    left_lock.lock();
                }
            }
        }
        left_son->working_flag = true;
        // left son is locked
        std::unique_lock<std::mutex> right_lock(right_son->mtx);
        if (right_son->rebuild_buffer_ptr != nullptr) {
            if (right_son->working_flag != false) {
                auto start = std::chrono::system_clock::now();
                while (right_son->working_flag != false) {
                    right_lock.unlock();
                    auto dur = std::chrono::duration_cast<std::chrono::microseconds> (
                        std::chrono::system_clock::now() - start
                    );
                    if (dur > 1ms) {
                        throw std::runtime_error("right son over when update");
                    }
                    right_lock.lock();
                }
            }
        }
        right_son->working_flag = true;

        node->tree_total_size = left_son->tree_total_size + right_son->tree_total_size + 1;
        node->tree_valid_size = left_son->tree_valid_size + right_son->tree_valid_size + (int)(!node->deleted);
        node->tree_deleted = (node->tree_valid_size == 0) ? true : false;
        node->valid_range = merge_range(left_son->valid_range, right_son->valid_range);
        if (node->deleted) {
            node->valid_range = merge_range(dummy_node_left->valid_range, node->valid_range);
        }
        else {
            node->valid_range = merge_range(node, node->valid_range);
        }
        left_son->working_flag = false;
        right_son->working_flag = false;
        left_lock.unlock();
        right_lock.unlock();

        // Check balance factor here.
        node->del_balance = static_cast<float>(node->tree_total_size - node->tree_valid_size) / node->tree_total_size;
        node->height_balance = static_cast<float>(left_son->tree_total_size) / (node->tree_total_size - 1.0); // always not div 0
        node->height_balance = std::max(node->height_balance, (1 - node->height_balance));
        node->balanced = (node->height_balance < height_criterion && node->del_balance < del_criterion);
        if (node->tree_total_size < 20) {
            node->balanced = true;
        }
    }

    // No need for current, but not for son
    void propagate(Node *node) {
        if (node == nullptr || !node->need_push_down) { return; }
        if (node->left_son != nullptr) {
            std::unique_lock<std::mutex> son_lock(node->left_son->mtx);
            if (node->left_son->rebuild_buffer_ptr != nullptr) {
                if (node->left_son->rebuild_buffer_ptr->st == RebuildBuffer::status::logging) {
                    OpContext op;
                    op.op = operation::push_down;
                    node->left_son->rebuild_buffer_ptr->logger.push(op);
                    son_lock.unlock();
                }
                else {
                    while (node->left_son->working_flag) {
                        son_lock.unlock();
                        std::this_thread::sleep_for(100us);
                        son_lock.lock();
                    }
                }
            }
            else {
                set_tree_del(node->left_son);
            }
        }
        if (node->right_son != nullptr) {
            std::unique_lock<std::mutex> son_lock(node->right_son->mtx);
            if (node->right_son->rebuild_buffer_ptr != nullptr) {
                if (node->right_son->rebuild_buffer_ptr->st == RebuildBuffer::status::logging) {
                    OpContext op;
                    op.op = operation::push_down;
                    node->right_son->rebuild_buffer_ptr->logger.push(op);
                    son_lock.unlock();
                }
                else {
                    while (node->right_son->working_flag) {
                        son_lock.unlock();
                        std::this_thread::sleep_for(100us);
                        son_lock.lock();
                    }
                }
            }
            else {
                set_tree_del(node->right_son);
            }
        }
        node->need_push_down = false;
    }


public:

    float dist_sq(Node *node, PointType &point) {
        return std::pow(node->x - point.x, 2) + std::pow(node->y - point.y, 2) + std::pow(node->z - point.z, 2);
    }

    // If the point inside range, ret 0, else nearest to range box.
    float min_box_dist_sq(Range &range, PointType &point) {
        using std::abs, std::min, std::pow;
        float rx, ry, rz;
        if (range[0][0] <= point.x && point.x <= range[0][1]) { rx = 0; }
        else { rx = min(abs(range[0][0] - point.x), abs(range[0][1] - point.x)); }

        if (range[1][0] <= point.y && point.y <= range[1][1]) { ry = 0; }
        else { ry = min(abs(range[1][0] - point.y), abs(range[1][1] - point.y)); }

        if (range[2][0] <= point.z && point.z <= range[2][1]) { rz = 0; }
        else { rz = min(abs(range[2][0] - point.z), abs(range[2][1] - point.z)); }

        return (rx * rx + ry * ry + rz * rz);
    }

    void set_tree_del(Node *node) {

        node->need_push_down = true;
        node->tree_deleted = true;  // always true;
        node->deleted = true;
        node->valid_range = dummy_node_left->valid_range;
        node->tree_total_size = node->tree_total_size;
        node->tree_valid_size = 0;
    }

    void init_dummy_node(void) {
        dummy_node_left = new Node;
        // The properties of this range:
        // All range operations containing: merge, compare
        // dummy_node_left->valid_range = {{FINF, -FINF}, {FINF, -FINF}, {FINF, -FINF}};
        dummy_node_left->tree_total_size = 0;
        dummy_node_left->tree_valid_size = 0;
        dummy_node_left->tree_deleted = true;
        dummy_node_left->deleted = true;
        dummy_node_right = new Node;

        Node::copy(dummy_node_right, dummy_node_left);
    }

    std::vector<std::vector<float>>
    merge_range(std::vector<std::vector<float>> &range1, std::vector<std::vector<float>> &range2) {
        std::vector<std::vector<float>> range (3, std::vector<float>(2, 0));
        range[0][0] = std::min(range1[0][0], range2[0][0]);
        range[0][1] = std::max(range1[0][1], range2[0][1]);
        range[1][0] = std::min(range1[1][0], range2[1][0]);
        range[1][1] = std::max(range1[1][1], range2[1][1]);
        range[2][0] = std::min(range1[2][0], range2[2][0]);
        range[2][1] = std::max(range1[2][1], range2[2][1]);
        return range;
    }

    Range merge_range(Node *node, Range &range) {
        Range ret(3, std::vector<float>(2, 0));
        ret[0][0] = std::min(range[0][0], node->x);
        ret[0][1] = std::max(range[0][1], node->x);
        ret[1][0] = std::min(range[1][0], node->y);
        ret[1][1] = std::max(range[1][1], node->y);
        ret[2][0] = std::min(range[2][0], node->z);
        ret[2][1] = std::max(range[2][1], node->z);
        return ret;
    }

    Range merge_range(PointType &point, Range &range) {
        Range ret(3, std::vector<float>(2, 0));
        ret[0][0] = std::min(range[0][0], point.x);
        ret[0][1] = std::max(range[0][1], point.x);
        ret[1][0] = std::min(range[1][0], point.y);
        ret[1][1] = std::max(range[1][1], point.y);
        ret[2][0] = std::min(range[2][0], point.z);
        ret[2][1] = std::max(range[2][1], point.z);
        return ret;

    }

    bool is_subset(Range &subset, Range &set) {
        return (set[0][0] <= subset[0][0] && set[0][1] >= subset[0][1] &&
            set[1][0] <= subset[1][0] && set[1][1] >= subset[1][1] &&
            set[2][0] <= subset[2][0] && set[2][1] >= subset[2][1]);
    }

    bool is_subset(Node *node, Range &set) {
        return (node->x >= set[0][0] && node->x <= set[0][1] &&
                node->y >= set[1][0] && node->y <= set[1][1] &&
                node->z >= set[2][0] && node->z <= set[2][1]);
    }

    bool is_subset(PointType &node, Range &set) {
        return (node.x >= set[0][0] && node.x <= set[0][1] &&
                node.y >= set[1][0] && node.y <= set[1][1] &&
                node.z >= set[2][0] && node.z <= set[2][1]);
    }

    bool is_intersect(Range &set1, Range &set2) {
        return !(set1[0][0] > set2[0][1] || set1[0][1] < set2[0][0] ||
            set1[1][0] > set2[1][1] || set1[1][1] < set2[1][0] ||
            set1[2][0] > set2[2][1] || set1[2][1] < set2[2][0]);
        }


    // heap
    Node* pnt_2_node(PointType &pnt) {
        Node *node = new Node;
        node->x = pnt.x;
        node->y = pnt.y;
        node->z = pnt.z;
        node->valid_range = {{node->x, node->x}, {node->y, node->y}, {node->z, node->z}};
        return node;
    }

    // stack
    PointType node_2_pnt(Node *node) {
        PointType p;
        p.x = node->x;
        p.y = node->y;
        p.z = node->z;
        return p;
    }

    double dist_node_pnt(Node *node, PointType &point) {
        double dist = std::pow(node->x - point.x, 2) + std::pow(node->y - point.y, 2) + std::pow(node->z - point.z, 2);
        return dist;
    }
public:
    Node* static_root = nullptr;
    std::mutex root_ptr_mtx;
    Node* root_ptr = nullptr;
    Node* dummy_node_left;
    Node* dummy_node_right;
    float del_criterion, height_criterion;

    PCLCmp<PointType> cmpx, cmpy, cmpz;

    RangeCmp cmp_down;
    RangeHash hash_down;
    HashQueue downsampling_register;
    std::vector<PointType> rebuild_buffer_main;  // main thread
    float voxel_size;
    int downsampling_requests = 0;
    int downsampling_executes = 0;

    std::atomic_bool terminate = false;
    RebuildBufferCmp cmp_rebuild;
    std::mutex rebuild_q_mtx;
    PriorityQueue<RebuildBuffer*, RebuildBufferCmp> rebuild_queue;
    std::thread second_th;
};


#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>

// TODO:
// re-organize push down and pull up, focusing on range, and status.


int main() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (float x = 0.21; x < 1.8; x+= 0.1) {
        for (float y = 0.21; y < 1.8; y += 0.1) {
            for (float z = 0.21; z < 1.8; z+= 0.1) { 
                pcl::PointXYZ p (x + (std::rand() % 100 - 50.) / 10000.,
                                 y + (std::rand() % 100 - 50.) / 10000.,
                                 z + (std::rand() % 100 - 50.) / 10000.);
                cloud->points.push_back(p);
            }
        }
    }
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
    cloud->resize(cloud->width * cloud->height);
    pcl::io::savePCDFile("../materials/box.pcd", *cloud, true);
    std::cout << "finish save" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../materials/test.pcd", *cloud2) == -1) {
        std::cout << "read failed" << std::endl;
    }
    IKDTree<pcl::PointXYZ> tree;
    std::vector<pcl::PointXYZ> vec;
    for (auto &i : cloud2->points) {
        vec.push_back(i);
    }
    tree.insert(vec, true, false);
    std::vector<std::vector<float>> range = {{{-1.76, 3.72}, {-1.13, 19.87}, {-5.57, 19.83}}};
    tree.box_delete(tree.root_ptr, range, true);

    // check knn search code
    pcl::PointXYZ target (0.5, 0.5, 0.5);
    tree.downsampling_execute();
    printf("Dsp requests: %d, exe %d, conflict %.3f \n", tree.downsampling_requests, tree.downsampling_executes, (float)tree.downsampling_register.get_conflict_cnt() / tree.downsampling_executes);
    std::vector<pcl::PointXYZ> find_points;
    std::vector<float> dist_sq;
    std::cout << "waiting for second" << std::endl;
    // std::this_thread::sleep_for(20s);
    tree.knn_search(tree.root_ptr, target, 7, 1, dist_sq, find_points);

    // tree.rebuild_subtree(tree.root_ptr);
    std::vector<pcl::PointXYZ> vec_of_node;
    std::this_thread::sleep_for(1s);
    tree.flatten(tree.root_ptr, vec_of_node);

    std::vector<float> dist;
    std::vector<pcl::PointXYZ> pnt;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis (new pcl::PointCloud<pcl::PointXYZRGB>);
    for (auto &i : vec_of_node) {
        pcl::PointXYZRGB p;
        p.x = i.x;
        p.y = i.y;
        p.z = i.z;
        p.r = 0;
        p.g = 0;
        p.b = 255;
        cloud_vis->points.push_back(p);
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr search(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (auto &i : find_points) {
        pcl::PointXYZRGB p;
        p.x = i.x;
        p.y = i.y;
        p.z =  i.z;
        p.r = 255;
        p.g = 0;
        p.b = 0;
        search->points.push_back(p);
    }
    pcl::PointXYZRGB t;
    t.x = 0.5;
    t.y = 0.5;
    t.z = 0.5;
    t.r = 0;
    t.g = 255;
    t.b = 0;
    search->points.push_back(t); 
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZRGB> field_color (cloud_vis, "z");
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud_vis, field_color, "raw cloud");
    // viewer->addPointCloud<pcl::PointXYZ> (cloud2, "raw raw");

    viewer->addPointCloud<pcl::PointXYZRGB>(search, "search cloud");
    viewer->addCoordinateSystem(1.0, "nnn");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "search cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "raw cloud");
    while (1) {
        viewer->spinOnce(1);
    }
}

