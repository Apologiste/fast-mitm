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

int rank, size;

typedef uint64_t u64;       /* portable 64-bit integer */
typedef uint32_t u32;       /* portable 32-bit integer */
struct __attribute__ ((packed)) entry { u32 k; u64 v; };  /* hash table entry */

/***************************** global variables ******************************/

u64 n = 0;         /* block size (in bits) */
u64 mask;          /* this is 2**n - 1 */

u64 dict_size;     /* number of slots in the hash table */
struct entry *A;   /* the hash table */

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

/******************************** SHARDED DICTIONARY  *******************************/


/*
 * Détermine quel processus MPI est responsable d’une key.
 * Elle renvoie le rang MPI qui doit stocker ou gérer cette key.
 * 
 *
 * @param : key – clé 64 bits.
 * @return : n entier dans [0, size-1] : le numéro du processus propriétaire.
 *
 */
static inline int owner_of(u64 key) {
    return murmur64(key) % size;
}

/*
 * Insère une paire (key → value) dans le dictionnaire distribué.
 *
 * Si le processus courant est le propriétaire de la clé (owner_of(key)),
 * l'insertion est faite localement via dict_insert().  
 * Sinon, la paire (key, value) est envoyée au processus propriétaire
 * via MPI_Send.
 *
 * @param key   Clé 64 bits à insérer.
 * @param value Valeur associée à la clé.
 *
 */
void dist_dict_insert(u64 key, u64 value) {
    int owner = owner_of(key);

    if (owner == rank) {
        dict_insert(key, value);
        return;
    }

    u64 msg[2] = {key, value};
    MPI_Send(msg, 2, MPI_UINT64_T, owner, 1, MPI_COMM_WORLD);
}


/*
 * Effectue une recherche (probe) dans le dictionnaire distribué.
 *
 * Si le processus courant est propriétaire de la clé (owner_of(key)),
 * la recherche est effectuée localement via dict_probe().
 * Sinon, le processus envoie une requête au propriétaire, qui effectue
 * la recherche puis renvoie :
 *    - le nombre de résultats,
 *    - les valeurs correspondantes.
 *
 * @param key     Clé 64 bits à rechercher.
 * @param maxval  Taille maximale du tableau values[].
 * @param values  Tableau où seront stockées les valeurs trouvées.
 *
 * @return  Nombre de valeurs trouvées (0, 1, …).  
 *          Retourne -1 si plus de maxval résultats existent.
 *
 */
int dist_dict_probe(u64 key, int maxval, u64 values[]) {
    int owner = owner_of(key);

    if (owner == rank) {
        return dict_probe(key, maxval, values);
    }

    /* Send request (only the key) */
    MPI_Send(&key, 1, MPI_UINT64_T, owner, 2, MPI_COMM_WORLD);

    /* Receive response count */
    int count = 0;
    MPI_Recv(&count, 1, MPI_INT, owner, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (count > maxval) return -1;

    /* Receive values */
    if (count > 0) {
        MPI_Recv(values, count, MPI_UINT64_T, owner, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return count;
}


/*
 * Boucle de service MPI : chaque processus gère les requêtes distantes.
 *
 * Cette boucle tourne en permanence et traite trois types de messages :
 *
 *   TAG 0 : ordre d'arrêt — le processus quitte la boucle.
 *   TAG 1 : insertion — reçoit (key, value) et appelle dict_insert().
 *   TAG 2 : recherche — reçoit une clé, effectue dict_probe(), puis renvoie
 *            le nombre de résultats et éventuellement la liste des valeurs.
 *
 * Le processus agit donc comme un "serveur" pour toutes les clés dont il est
 * propriétaire, en répondant aux insertions et probes envoyés par les autres rangs.
 *
 */
void dict_worker_loop() {
    MPI_Status st;

    while (1) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);

        if (st.MPI_TAG == 0) {
            MPI_Recv(NULL, 0, MPI_BYTE, st.MPI_SOURCE, 0, MPI_COMM_WORLD, &st);
            return; /* stop request */
        }

        if (st.MPI_TAG == 1) {
            /* Insert (key,value) locally */
            u64 msg[2];
            MPI_Recv(msg, 2, MPI_UINT64_T, st.MPI_SOURCE, 1, MPI_COMM_WORLD, &st);
            dict_insert(msg[0], msg[1]);
        }

        if (st.MPI_TAG == 2) {
            /* Probe request */
            u64 key;
            MPI_Recv(&key, 1, MPI_UINT64_T, st.MPI_SOURCE, 2, MPI_COMM_WORLD, &st);

            u64 buf[256];
            int count = dict_probe(key, 256, buf);

            MPI_Send(&count, 1, MPI_INT, st.MPI_SOURCE, 3, MPI_COMM_WORLD);

            if (count > 0) {
                MPI_Send(buf, count, MPI_UINT64_T, st.MPI_SOURCE, 4, MPI_COMM_WORLD);
            }
        }
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
    double start = wtime();
    u64 N = 1ull << n;

    
    for (u64 x = 0; x < N; x++) {
        u64 z = f(x);
        dist_dict_insert(z, x); // insertion locale pour le propriétaire de la clé
    }

    double mid = wtime();
    printf("Fill: %.1fs\n", mid - start);
    
    int nres = 0;
    u64 ncandidates = 0;
    u64 x[256];
    for (u64 z = 0; z < N; z++) {
        u64 y = g(z);
        int nx = dist_dict_probe(y, 256, x); // on cherche le propriétaire de la clé
        //assert(nx >= 0);
        ncandidates += nx;
        for (int i = 0; i < nx; i++)
            if (is_good_pair(x[i], z)) {
            	if (nres == maxres)
            		return -1;
            	k1[nres] = x[i];
            	k2[nres] = z;
            	printf("SOLUTION FOUND!\n");
            	nres += 1;
            }
    }
    printf("Probe: %.1fs. %" PRId64 " candidate pairs tested\n", wtime() - mid, ncandidates);
    return nres;
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
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


	process_command_line_options(argc, argv);
    printf("Running with n=%d, C0=(%08x, %08x) and C1=(%08x, %08x)\n", 
        (int) n, C[0][0], C[0][1], C[1][0], C[1][1]);


	dict_setup(1.125 * ((1ull << n) / size)); //dictionnaire local


    if (rank != 0) {
        dict_worker_loop(); //serveur de réponses pour processus
        MPI_Finalize();
        return 0;
    }

    //chaque processus cherche des collisions
    u64 k1[16], k2[16];
    int nkey = golden_claw_search(16, k1, k2);

    //fin de travail, 
    for (int r = 1; r < size; r++){
        MPI_Send(NULL, 0, MPI_BYTE, r, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();


}