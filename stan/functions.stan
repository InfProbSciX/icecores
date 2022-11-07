functions {
    real hmm_marginal_banded(matrix log_omega,
                         vector Gamma_diag,
                         vector rho) {
        int K = dims(log_omega)[1];
        int N = dims(log_omega)[2];

        vector[K] log_alpha;
        vector[K] inner_sum;
        vector[2] inner_vec;
        vector[K] log_Gamma_diag = log(Gamma_diag);
        vector[K] log_1mGamma_diag = log1m(Gamma_diag);

        int min_i; int max_i;

        log_alpha = log_omega[, 1] + log(rho);

        if (N > 1) {
            for (n in 2:N) {
                for (i in 1:K) {
                    if (i == 1) {
                        inner_sum[i] = log_alpha[i] + log_Gamma_diag[i];
                    } else {
                        inner_vec[1] = log_alpha[i - 1] + log_1mGamma_diag[i - 1];
                        inner_vec[2] = log_alpha[i] + log_Gamma_diag[i];
                        inner_sum[i] = log_sum_exp(inner_vec);
                    }
                }
                log_alpha = log_omega[, n] + inner_sum;
            }
        }

        return log_sum_exp(log_alpha);
    }

    matrix diag_trans_to_full(vector trans_mat_diag) {
        int n = size(trans_mat_diag);
        matrix[n, n] full_mat = diag_matrix(trans_mat_diag);

        for (i in 1:(n - 1)) {
            full_mat[i, i + 1] = 1 - trans_mat_diag[i];
        }
        full_mat[n, n] = 1.0 - 1e-6;
        full_mat[n, 1] = 1e-6;

        return full_mat;
    }

    vector tile(vector x, int r) {
        int n = size(x);
        vector[n * r] result;
        for (i in 1:r) {
            result[((i - 1)*n + 1):(i*n)] = x;
        }
        return result;
    }
}
