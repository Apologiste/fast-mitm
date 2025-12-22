#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>
#include <err.h>
#include <assert.h>

#include <mpi.h>
#include <omp.h>
#include <time.h>

typedef uint64_t u64;       /* portable 64-bit integer */
typedef uint32_t u32;       /* portable 32-bit integer */
struct __attribute__ ((packed)) entry { u32 k; u64 v; };  /* hash table entry */

struct key_value {u64 key; u64 value;};

/***************************** global variables ******************************/

int rank, world_size;   /* utilisé par MPI */

u64 n = 0;         /* block size (in bits) */
u64 mask;          /* this is 2**n - 1 */

u64 dict_size;     /* number of slots in the hash table */
struct entry *A;   /* the hash table */

struct entry *local_A;  /* dictionnaire local */
u64 local_dict_size;    /* taille du dictionnaire local */

/* (P, C) : two plaintext-ciphertext pairs */
u32 P[2][2] = {{0, 0}, {0xffffffff, 0xffffffff}};
u32 C[2][2];

/************************ tools and utility functions *************************/

double wtime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (double)ts.tv_sec + ts.tv_usec / 1E6;
}

// murmur64 hash functions, tailorized for 64-bit ints / Cf. Daniel Lemire
u64 murmur64(u64 x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

/* represent n in 4 bytes */
void human_format(u64 n, char *target)
{
    if (n < 1000) {
        sprintf(target, "%" PRId64, n);
        return;
    }
    if (n < 1000000) {
        sprintf(target, "%.1fK", n / 1e3);
        return;
    }
    if (n < 1000000000) {
        sprintf(target, "%.1fM", n / 1e6);
        return;
    }
    if (n < 1000000000000ll) {
        sprintf(target, "%.1fG", n / 1e9);
        return;
    }
    if (n < 1000000000000000ll) {
        sprintf(target, "%.1fT", n / 1e12);
        return;
    }
}

/******************************** SPECK block cipher **************************/

#define ROTL32(x,r) (((x)<<(r)) | (x>>(32-(r))))
#define ROTR32(x,r) (((x)>>(r)) | ((x)<<(32-(r))))

#define ER32(x,y,k) (x=ROTR32(x,8), x+=y, x^=k, y=ROTL32(y,3), y^=x)
#define DR32(x,y,k) (y^=x, y=ROTR32(y,3), x^=k, x-=y, x=ROTL32(x,8))

void Speck64128KeySchedule(const u32 K[],u32 rk[])
{
    u32 i,D=K[3],C=K[2],B=K[1],A=K[0];
    for(i=0;i<27;){
        rk[i]=A; ER32(B,A,i++);
        rk[i]=A; ER32(C,A,i++);
        rk[i]=A; ER32(D,A,i++);
    }
}

void Speck64128Encrypt(const u32 Pt[], u32 Ct[], const u32 rk[])
{
    u32 i;
    Ct[0]=Pt[0]; Ct[1]=Pt[1];
    for(i=0;i<27;)
        ER32(Ct[1],Ct[0],rk[i++]);
}

void Speck64128Decrypt(u32 Pt[], const u32 Ct[], u32 const rk[])
{
    int i;
    Pt[0]=Ct[0]; Pt[1]=Ct[1];
    for(i=26;i>=0;)
        DR32(Pt[1],Pt[0],rk[i--]);
}

/******************************** dictionary ********************************/

/*
 * "classic" hash table for 64-bit key-value pairs, with linear probing.  
 * It operates under the assumption that the keys are somewhat random 64-bit integers.
 * The keys are only stored modulo 2**32 - 5 (a prime number), and this can lead 
 * to some false positives.
 */
static const u32 EMPTY = 0xffffffff;
static const u64 PRIME = 0xfffffffb;

/* allocate a hash table with `size` slots (12*size bytes) */
void dict_setup(u64 size)
{
	dict_size = size;
	char hdsize[8];
	human_format(dict_size * sizeof(*A), hdsize);
	printf("Dictionary size: %sB\n", hdsize);

	A = malloc(sizeof(*A) * dict_size);
	if (A == NULL)
		err(1, "impossible to allocate the dictionnary");
	for (u64 i = 0; i < dict_size; i++)
		A[i].k = EMPTY;
}

/* version pour un sharded dictionnary */
void shard_dict_setup(u64 local_size) {

    // Chaque dictionnaire local est plus petit
    local_dict_size = (local_size / world_size) * 2;
    if (local_dict_size == 0){
        local_dict_size = 1; //sécurité
    }


    local_A = malloc(local_dict_size * sizeof(struct entry));
    if (!local_A) {
        err(1, "cannot allocate local shard");
    }

    #pragma omp parallel for //FIXME
    for (u64 i = 0; i < local_dict_size; i++) {
        local_A[i].k = EMPTY;
    }
}

/* Insert the binding key |----> value in the dictionnary */
void dict_insert(u64 key, u64 value)
{
    u64 h = murmur64(key) % dict_size;
    for (;;) {
        if (A[h].k == EMPTY)
            break;
        h += 1;
        if (h == dict_size)
            h = 0;
    }
    assert(A[h].k == EMPTY);
    A[h].k = key % PRIME;
    A[h].v = value;
}

/* version pour un sharded dictionnary */
void shard_dict_insert(u64 key, u64 value) {
    u64 h = murmur64(key) % local_dict_size;

    for (;;) {
        if (local_A[h].k == EMPTY)
            break;

        h++;
        if (h == local_dict_size)
            h = 0;
    }

    local_A[h].k = key % PRIME;
    local_A[h].v = value;
}

/* Query the dictionnary with this `key`.  Write values (potentially) 
 *  matching the key in `values` and return their number. The `values`
 *  array must be preallocated of size (at least) `maxval`.
 *  The function returns -1 if there are more than `maxval` results.
 */
int dict_probe(u64 key, int maxval, u64 values[])
{
    u32 k = key % PRIME;
    u64 h = murmur64(key) % dict_size;
    int nval = 0;
    for (;;) {
        if (A[h].k == EMPTY)
            return nval;
        if (A[h].k == k) {
        	if (nval == maxval)
        		return -1;
            values[nval] = A[h].v;
            nval += 1;
        }
        h += 1;
        if (h == dict_size)
            h = 0;
   	}
}

/* version pour un sharded dictionnary */
int shard_probe_local(u64 key, int maxval, u64 values[]) {
    u32 k = key % PRIME;
    u64 h = murmur64(key) % local_dict_size;

    int nval = 0;

    for (;;) {
        if (local_A[h].k == EMPTY)
            return nval;

        if (local_A[h].k == k) {
            if (nval == maxval)
                return -1;
            values[nval] = local_A[h].v;
            nval++;
        }

        h++;
        if (h == local_dict_size)
            h = 0;
    }
}

/***************************** MITM problem ***********************************/

/* f : {0, 1}^n --> {0, 1}^n.  Speck64-128 encryption of P[0], using k */
u64 f(u64 k)
{
    assert((k & mask) == k);
    u32 K[4] = {k & 0xffffffff, k >> 32, 0, 0};
    u32 rk[27];
    Speck64128KeySchedule(K, rk);
    u32 Ct[2];
    Speck64128Encrypt(P[0], Ct, rk);
    return ((u64) Ct[0] ^ ((u64) Ct[1] << 32)) & mask;
}

/* g : {0, 1}^n --> {0, 1}^n.  speck64-128 decryption of C[0], using k */
u64 g(u64 k)
{
    assert((k & mask) == k);
    u32 K[4] = {k & 0xffffffff, k >> 32, 0, 0};
    u32 rk[27];
    Speck64128KeySchedule(K, rk);
    u32 Pt[2];
    Speck64128Decrypt(Pt, C[0], rk);
    return ((u64) Pt[0] ^ ((u64) Pt[1] << 32)) & mask;
}

bool is_good_pair(u64 k1, u64 k2)
{
    u32 Ka[4] = {k1 & 0xffffffff, k1 >> 32, 0, 0};
    u32 Kb[4] = {k2 & 0xffffffff, k2 >> 32, 0, 0};
    u32 rka[27];
    u32 rkb[27];
    Speck64128KeySchedule(Ka, rka);
    Speck64128KeySchedule(Kb, rkb);
    u32 mid[2];
    u32 Ct[2];
    Speck64128Encrypt(P[1], mid, rka);
    Speck64128Encrypt(mid, Ct, rkb);
    return (Ct[0] == C[1][0]) && (Ct[1] == C[1][1]);
}

/******************************************************************************/

/* search the "golden collision" */
int golden_claw_search(int maxres, u64 k1[], u64 k2[])
{

    MPI_Datatype MPI_KEYVALUE;
    MPI_Type_contiguous(2, MPI_UINT64_T, &MPI_KEYVALUE);
    MPI_Type_commit(&MPI_KEYVALUE);

    double start = wtime();
    u64 N = 1ull << n;

    struct key_value **kv_to_send = malloc(world_size*sizeof(struct key_value *));
    // Allouer le tableau de tableaux
    for(int i = 0 ; i<world_size ; i++){
        kv_to_send[i] = malloc(2*(N/world_size)*sizeof(struct key_value));
    }


    /*******************************************************************
    *  PHASE 1 : remplir le dictionnaire f(x) -> z
    ******************************************************************/

    #define BUF_SIZE 4096  // taille du buffer d’envoi

    int *recv_counts = calloc(world_size, sizeof(int));

    int *send_counts = calloc(world_size, sizeof(int));
    MPI_Request *send_requests = malloc(world_size * sizeof(MPI_Request));
    MPI_Request *recv_requests = malloc(world_size * sizeof(MPI_Request));
    struct key_value **recv_buffers = malloc(world_size * sizeof(struct key_value *));

    #pragma omp parallel
    {
        struct key_value *buf[world_size];
        int buf_counts[world_size];

        // initialisation buffers locaux par thread
        for(int i=0;i<world_size;i++){
            buf[i] = malloc(BUF_SIZE*sizeof(struct key_value));
            buf_counts[i] = 0;
        }

        #pragma omp for schedule(static)
        for(u64 x = rank; x < N; x += world_size){
            u64 z = f(x);
            int shard_id = z % world_size;

            if(shard_id == rank){
                shard_dict_insert(z, x);
            } else {
                buf[shard_id][buf_counts[shard_id]].key = z;
                buf[shard_id][buf_counts[shard_id]].value = x;
                buf_counts[shard_id]++;

                // buffer plein : envoyer au shard propriétaire
                if(buf_counts[shard_id] == BUF_SIZE){
                    #pragma omp critical
                    {
                        MPI_Isend(buf[shard_id], BUF_SIZE, MPI_KEYVALUE, shard_id, 1, MPI_COMM_WORLD, &send_requests[shard_id]);
                        send_counts[shard_id] += BUF_SIZE;
                    }
                    buf_counts[shard_id] = 0;
                }
            }
        }

        // envoyer les éléments restants du buffer
        #pragma omp critical
        for(int i=0;i<world_size;i++){
            if(buf_counts[i] > 0){
                MPI_Isend(buf[i], buf_counts[i], MPI_KEYVALUE, i, 1, MPI_COMM_WORLD, &send_requests[i]);
                send_counts[i] += buf_counts[i];
            }
            free(buf[i]);
        }
    }

    // préparation des réceptions
    for(int i=0;i<world_size;i++){
        if(i == rank){
            recv_buffers[i] = NULL;
            recv_requests[i] = MPI_REQUEST_NULL;
        } else {
            recv_buffers[i] = malloc(N/world_size * sizeof(struct key_value)); // dimension maximale possible
            MPI_Irecv(recv_buffers[i], N/world_size, MPI_KEYVALUE, i, 1, MPI_COMM_WORLD, &recv_requests[i]);
        }
    }

    // attendre et insérer les données reçues
    for(int i=0;i<world_size;i++){
        if(recv_requests[i] != MPI_REQUEST_NULL){
            MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);

            // insérer dans le dictionnaire local
            for(int j=0;j<N/world_size;j++){
                if(recv_buffers[i][j].key == 0 && recv_buffers[i][j].value == 0)
                    break; // fin des données valides
                shard_dict_insert(recv_buffers[i][j].key, recv_buffers[i][j].value);
            }
            free(recv_buffers[i]);
        }

        if(send_requests[i] != MPI_REQUEST_NULL){
            MPI_Wait(&send_requests[i], MPI_STATUS_IGNORE);
        }
    }

    free(send_requests);
    free(recv_requests);
    free(recv_buffers);
    free(send_counts);
    free(recv_counts);

    MPI_Barrier(MPI_COMM_WORLD);


    /*******************************************************************
     *  PHASE 2 : probing g(z) -> y et test des collisions
    ******************************************************************/

    send_counts = calloc(world_size, sizeof(int));
    recv_counts = calloc(world_size, sizeof(int));

    //requests = malloc(world_size*sizeof(MPI_Request));

    u64 local_k1[16], local_k2[16];
    u64 nres_local = 0;
    u64 x_local[256];

    // -------- boucle d’envoi (g)--------
    #pragma omp parallel
    {
        int tid = omp_get_thread_num(); //numero thread
        int nthreads = omp_get_num_threads(); //nb total de thread
        struct key_value *kv_local = malloc(world_size * 2 * (N/world_size/nthreads + 1) * sizeof(struct key_value)); //buffer local par thread des k-v qu'il doit envoyer aux autres
        int *local_counts = calloc(world_size, sizeof(int)); //buffer du thread

        #pragma omp for schedule(static)
        for (u64 z = rank; z < N; z += world_size) {
            u64 y = g(z);
            int shard_id = y % world_size;

            if (shard_id == rank) {
                int nx = shard_probe_local(y, 256, x_local);
                for (int i = 0; i < nx; i++) {
                    if (nres_local >= maxres) break;
                    if (is_good_pair(x_local[i], z)) {
                        #pragma omp critical
                        {
                            local_k1[nres_local] = x_local[i];
                            local_k2[nres_local] = z;
                            nres_local++;
                        }
                    }
                }
            } else {
                int idx = local_counts[shard_id];
                local_counts[shard_id]++;
                kv_local[shard_id*(N/world_size/nthreads + 1) + idx].key = y;
                kv_local[shard_id*(N/world_size/nthreads + 1) + idx].value = z;
            }
        }

        // fusionner les buffers locaux de chaque thread dans le buffer global
        #pragma omp critical
        for (int i=0; i<world_size; i++){
            for (int j=0; j<local_counts[i]; j++){
                kv_to_send[i][send_counts[i]++] = kv_local[i*(N/world_size/nthreads + 1) + j];
            }
        }

        free(kv_local);
        free(local_counts);
    }


    /*
    for(int i=0 ; i<world_size ; i++){
        MPI_Isend(kv_to_send[i], send_counts[i], MPI_KEYVALUE, i, 2, MPI_COMM_WORLD, &requests[i]);
    }


    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    
    // -------- receptions --------
    for(int i = 0 ; i<world_size ; i++) {
        if(recv_counts[i]){
            struct key_value *buffer = malloc((N/world_size)*sizeof(struct key_value));

            //MPI_Recv(buffer, recv_counts[i], MPI_KEYVALUE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Irecv(buffer, recv_counts[i], MPI_KEYVALUE, i, 2, MPI_COMM_WORLD, &requests[i]);
            for(int j = 0 ; j<recv_counts[i] && nres_local<maxres ; j++){

                int nx = shard_probe_local(buffer[j].key, 256, x_local);

                for (int i = 0; i < nx && nres_local < maxres; i++) {
                    if (is_good_pair(x_local[i], buffer[j].value)) {
                        local_k1[nres_local] = x_local[i];
                        local_k2[nres_local] = buffer[j].value;
                        nres_local++;
                    }
                }
            }
            free(buffer);
        }
    }

    // attendre les envois
    for(int i=0 ; i<world_size ; i++){
        MPI_Wait(&requests[i], MPI_STATUSES_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    
    for(int i = 0 ; i<world_size ; i++){
        free(kv_to_send[i]);
    }

    free(kv_to_send);
    free(send_counts);
    free(recv_counts);
    free(requests);
    */
    
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    send_requests = malloc(world_size * sizeof(MPI_Request));
    recv_requests = malloc(world_size * sizeof(MPI_Request));

    recv_buffers = malloc(world_size * sizeof(struct key_value *));

    // chaque processus s'annonce prêt à recevoir
    for(int i = 0; i < world_size; i++) {
        if(recv_counts[i] > 0) {
            recv_buffers[i] = malloc(recv_counts[i] * sizeof(struct key_value));
            // commence une réception asynchrone du processus i
            MPI_Irecv(recv_buffers[i], recv_counts[i], MPI_KEYVALUE, i, 2, MPI_COMM_WORLD, &recv_requests[i]);
        } else {
            // cas ou donnée à recevoir de ce processus
            recv_buffers[i] = NULL;
            recv_requests[i] = MPI_REQUEST_NULL; 
        }
    }

    // chaque processus envoie les couples aux propriétaires
    for(int i = 0; i < world_size; i++) {
        if(send_counts[i] > 0) {
            // envoi asynchrone vers le processus i
            // kv_to_send[i] : couples destinés au processus i
            MPI_Isend(kv_to_send[i], send_counts[i], MPI_KEYVALUE, i, 2, MPI_COMM_WORLD, &send_requests[i]);
        } else {
            // cas ou donnée à envoyer à ce processus
            send_requests[i] = MPI_REQUEST_NULL; 
        }
    }


    // On attend que toutes les réceptions soient terminées avant de traiter les données
    for(int i = 0; i < world_size; i++) {
        if(recv_requests[i] != MPI_REQUEST_NULL) {
            //bloque jusqu'à ce que la réception du processus i soit terminée
            MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
            
            // 
            for(int j = 0; j < recv_counts[i]; j++) {
                //shard_dict_insert(recv_buffers[i][j].key, recv_buffers[i][j].value);

                int nx = shard_probe_local(recv_buffers[i][j].key, 256, x_local);

                for (int k = 0; k < nx && nres_local < maxres; k++) {
                    if (is_good_pair(x_local[k], recv_buffers[i][j].value)) {
                        local_k1[nres_local] = x_local[k];
                        local_k2[nres_local] = recv_buffers[i][j].value;
                        nres_local++;
                    }
                }

            }
            free(recv_buffers[i]);
        }
    }

    // on attend que tous les envois soient terminés
    for(int i = 0; i < world_size; i++) {
        if(send_requests[i] != MPI_REQUEST_NULL) {
            MPI_Wait(&send_requests[i], MPI_STATUS_IGNORE);
        }
    }

    free(send_requests);
    free(recv_requests);
    free(recv_buffers);
    free(send_counts);
    free(recv_counts);
    free(kv_to_send);

    MPI_Barrier(MPI_COMM_WORLD);

    /*******************************************************************
     *  PHASE 3 : GATHER DES RÉSULTATS
    ******************************************************************/

    u64 all_nres[world_size];
    MPI_Gather(&nres_local, 1, MPI_UINT64_T,all_nres, 1, MPI_UINT64_T,0, MPI_COMM_WORLD);

    if (rank == 0) {

        u64 total = 0;
        for (int i = 0; i < world_size; i++)
            total += all_nres[i];

        u64 final = (total < maxres ? total : maxres);

        int *counts = malloc(world_size * sizeof(int));
        int *offset = malloc(world_size * sizeof(int));

        offset[0] = 0;
        for (int i = 0; i < world_size; i++) {
            counts[i] = all_nres[i];
            if (i > 0)
                offset[i] = offset[i - 1] + counts[i - 1];
        }

        u64 *global_k1 = malloc(total * sizeof(u64));
        u64 *global_k2 = malloc(total * sizeof(u64));

        MPI_Gatherv(local_k1, nres_local, MPI_UINT64_T,global_k1, counts, offset, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        MPI_Gatherv(local_k2, nres_local, MPI_UINT64_T,global_k2, counts, offset, MPI_UINT64_T,0, MPI_COMM_WORLD);

        for (u64 i = 0; i < final; i++) {
            k1[i] = global_k1[i];
            k2[i] = global_k2[i];
        }

        free(global_k1);
        free(global_k2);
        free(counts);
        free(offset);

        return final;
    }

    // autres ranks
    MPI_Gatherv(local_k1, nres_local, MPI_UINT64_T,NULL, NULL, NULL, MPI_UINT64_T,0, MPI_COMM_WORLD);

    MPI_Gatherv(local_k2, nres_local, MPI_UINT64_T,NULL, NULL, NULL, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    return 0;
}


/************************** command-line options ****************************/

void usage(char **argv)
{
        printf("%s [OPTIONS]\n\n", argv[0]);
        printf("Options:\n");
        printf("--n N                       block size [default 24]\n");
        printf("--C0 N                      1st ciphertext (in hex)\n");
        printf("--C1 N                      2nd ciphertext (in hex)\n");
        printf("\n");
        printf("All arguments are required\n");
        exit(0);
}

void process_command_line_options(int argc, char ** argv)
{
        struct option longopts[4] = {
                {"n", required_argument, NULL, 'n'},
                {"C0", required_argument, NULL, '0'},
                {"C1", required_argument, NULL, '1'},
                {NULL, 0, NULL, 0}
        };
        char ch;
        int set = 0;
        while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
                switch (ch) {
                case 'n':
                        n = atoi(optarg);
                        mask = (1ull << n) - 1;
                        break;
                case '0':
                        set |= 1;
                        u64 c0 = strtoull(optarg, NULL, 16);
                        C[0][0] = c0 & 0xffffffff;
                        C[0][1] = c0 >> 32;
                        break;
                case '1':
                        set |= 2;
                        u64 c1 = strtoull(optarg, NULL, 16);
                        C[1][0] = c1 & 0xffffffff;
                        C[1][1] = c1 >> 32;
                        break;
                default:
                        errx(1, "Unknown option\n");
                }
        }
        if (n == 0 || set != 3) {
        	usage(argv);
        	exit(1);
        }
}

/******************************************************************************/

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Le support multi-thread MPI n'est pas suffisant\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    process_command_line_options(argc, argv);

    if (rank == 0) {
        printf("Nombre de processus MPI : %d\n", world_size);
        #pragma omp parallel
    {
        #pragma omp single
        {
            int nthreads = omp_get_num_threads();
            printf("Nombre de threads OpenMP : %d\n", nthreads);
        }
    }
        printf("Running with n=%d, C0=(%08x, %08x) and C1=(%08x, %08x)\n", 
            (int) n, C[0][0], C[0][1], C[1][0], C[1][1]);
    }

    // Initialisation du dictionnaire shardé
    shard_dict_setup(1.125 * (1ull << n));

    u64 k1[16], k2[16];

    ///////////////EXECUTION/////////////////////
    double start, end;

    // Synchronisation avant le calcul
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    int nkey = golden_claw_search(16, k1, k2);

    // Synchronisation après le calcul
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    /////////////////////////////////////////////

    MPI_Finalize();

    // Rank 0 valide et affiche
    if (rank == 0) {
        if (nkey == 0) {
            printf("Aucune solution trouvée pour n=%lu\n", n);
            return 0;
        }


        for (int i = 0; i < nkey; i++) {
            assert(f(k1[i]) == g(k2[i]));
            assert(is_good_pair(k1[i], k2[i]));        
            printf("Solution trouvée: (%" PRIx64 ", %" PRIx64 ") [checked OK]\n",
                   k1[i], k2[i]);
        }

        printf("Temps d'exécution MPI : %.6f sec\n",
               end - start);
    }

    return 0;
}