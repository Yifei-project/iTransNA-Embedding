using DynamicalSystems
using Distances
using Statistics

function perform_single_evaluation(Y_ref::Dataset, Y::Dataset;
                        ε::Real = 0.05, w::Int = 1, lmin::Int = 2, kNN::Int = 1, z_est::Real = -1, w_est::Real = 1)
    N, L = size(Y)
    N_, L_ = size(Y_ref)
    # if L > 5
    #     metric = Chebyshev()
    # else
    #     metric = Euclidean() 
    # end
    # if L_ > 5
    #     metric_ref = Chebyshev()
    # else
    #     metric_ref = Euclidean() 
    # end
    R_ref = RecurrenceMatrix(Y_ref[1:N,:], GlobalRecurrenceRate(ε); metric = Euclidean())
    R = RecurrenceMatrix(Y[1:N,:], GlobalRecurrenceRate(ε); metric = Euclidean())

    f_ = jrp_rr_frac(R_ref, R)
    mfnn_ = mfnn(Y_ref[1:N,:], Y[1:N,:]; w = w, kNN = kNN)

    RQA_ref = rqa(R_ref; theiler = w, lmin = lmin)
    RQA = rqa(R; theiler = w, lmin = lmin)

    dim_ref = grassberger_proccacia_dim(Y_ref, estimate_boxsizes(Y_ref; z=-1, w=1); w=w, show_progress = false, norm=Chebyshev())
    dim = grassberger_proccacia_dim(Y, estimate_boxsizes(Y; z=z_est, w=w_est); w=w, show_progress = false, norm=Chebyshev())

    return mfnn_, f_, RQA_ref, RQA, dim_ref, dim

end

function output_metric_scores(Y_ref::Dataset, Y::Dataset;
                        ε::Real = 0.05, w::Int = 1, lmin::Int = 2, kNN::Int = 1, z_est::Real = -1, w_est::Real = 1)

    MFNN_score, JRRF_score, RQA_ref, RQA, d_ref, d = perform_single_evaluation(Y_ref, Y; ε, w, lmin, kNN, z_est, w_est)
    DET_score = 1 - abs(RQA_ref[:DET] - RQA[:DET])/RQA_ref[:DET]
    ENTR_score = 1 - abs(RQA_ref[:ENTR] - RQA[:ENTR])/RQA_ref[:ENTR]
    RTE_score = 1 - abs(RQA_ref[:RTE] - RQA[:RTE])/RQA_ref[:RTE]
    DIM_score = 1 - abs(d_ref - d)/d_ref
    records = zeros(4)
    records_ref = zeros(4) 
    records[1] = d
    records_ref[1] = d_ref
    records[2] = RQA[:DET]
    records_ref[2] = RQA_ref[:DET]
    records[3] = RQA[:ENTR]
    records_ref[3] = RQA_ref[:ENTR]
    records[4] = RQA[:RTE]
    records_ref[4] = RQA_ref[:RTE]

    return MFNN_score, DIM_score, JRRF_score, DET_score, ENTR_score, RTE_score, records, records_ref 
    
end

function perform_multiple_evaluation(Y_ref::Dataset, Y₁::Dataset,
                        Y₂::Dataset, Y₃::Dataset, Y₄::Dataset;
                        ε::Real = 0.05, w::Int = 1, lmin::Int = 2, kNN::Int = 1)

    N1 = length(Y₁)
    N2 = length(Y₂)
    N3 = length(Y₃)
    N4 = length(Y₄)
    N = minimum(hcat(N1, N2, N3, N4))

    R_ref = RecurrenceMatrix(Y_ref[1:N,:], GlobalRecurrenceRate(ε); metric = "euclidean")
    R1 = RecurrenceMatrix(Y₁[1:N,:], GlobalRecurrenceRate(ε))
    R2 = RecurrenceMatrix(Y₂[1:N,:], GlobalRecurrenceRate(ε))
    R3 = RecurrenceMatrix(Y₃[1:N,:], GlobalRecurrenceRate(ε))
    R4 = RecurrenceMatrix(Y₄[1:N,:], GlobalRecurrenceRate(ε))

    f1 = jrp_rr_frac(R_ref, R1)
    f2 = jrp_rr_frac(R_ref, R2)
    f3 = jrp_rr_frac(R_ref, R3)
    f4 = jrp_rr_frac(R_ref, R4)

    mfnn1 = mfnn(Y_ref[1:N,:], Y₁[1:N,:]; w = w, kNN = kNN)
    mfnn2 = mfnn(Y_ref[1:N,:], Y₂[1:N,:]; w = w, kNN = kNN)
    mfnn3 = mfnn(Y_ref[1:N,:], Y₃[1:N,:]; w = w, kNN = kNN)
    mfnn4 = mfnn(Y_ref[1:N,:], Y₄[1:N,:]; w = w, kNN = kNN)

    RQA_ref = rqa(R_ref; theiler = w, lmin = lmin)
    RQA1 = rqa(R1; theiler = w, lmin = lmin)
    RQA2 = rqa(R2; theiler = w, lmin = lmin)
    RQA3 = rqa(R3; theiler = w, lmin = lmin)
    RQA4 = rqa(R4; theiler = w, lmin = lmin)

    return mfnn1, mfnn2, mfnn3, mfnn4, f1, f2, f3, f4, RQA_ref, RQA1, RQA2,
                                            RQA3, RQA4, R_ref, R1, R2, R3, R4
end



"""
Computes the mututal false nearest neighbours (mfnn) for a reference trajectory
`Y_ref` and a reconstruction `Y_rec` after [^Rulkov1995].

Keyword arguments:

*`w = 1`: Theiler window for the surpression of serially correlated neighbors in
    the nearest neighbor-search
*`kNN = 1`: The number of considered nearest neighbours (in the paper always 1)

[^Rulkov1995]: Rulkov, Nikolai F. and Sushchik, Mikhail M. and Tsimring, Lev S. and Abarbanel, Henry D.I. (1995). [Generalized synchronization of chaos in directionally coupled chaotic systems. Physical Review E 51, 980](https://doi.org/10.1103/PhysRevE.51.980).
"""
function mfnn(Y_ref::Dataset, Y_rec::Dataset; w::Int = 1, kNN::Int = 1)

    @assert length(Y_ref) == length(Y_rec)
    @assert kNN > 0
    N = length(Y_ref)
    metric = Euclidean()

    # compute nearest neighbor distances for both trajectories
    vtree = KDTree(Y_ref, metric)
    allNNidxs_ref, _ = DelayEmbeddings.all_neighbors(vtree, Y_ref,
                                                        1:length(Y_ref), kNN, w)
    vtree = KDTree(Y_rec, metric)
    allNNidxs_rec, _ = DelayEmbeddings.all_neighbors(vtree, Y_rec,
                                                        1:length(Y_rec), kNN, w)

    F = zeros(N)
    factor1_nom = zeros(kNN)
    factor1_denom = zeros(kNN)
    factor2_nom = zeros(kNN)
    factor2_denom = zeros(kNN)
    for i = 1:N
        for j = 1:kNN
            factor1_nom[j] = evaluate(Euclidean(), Y_rec[i], Y_rec[allNNidxs_ref[i][j]])
            factor1_denom[j] = evaluate(Euclidean(), Y_ref[i], Y_ref[allNNidxs_ref[i][j]])
            factor2_nom[j] = evaluate(Euclidean(), Y_ref[i], Y_ref[allNNidxs_rec[i][j]])
            factor2_denom[j] = evaluate(Euclidean(), Y_rec[i], Y_rec[allNNidxs_rec[i][j]])
        end
        factor1 = sum(factor1_nom)/sum(factor2_denom)
        factor2 = sum(factor2_nom)/sum(factor1_denom)
        F[i] = factor1*factor2                                         # Eq.(27)
    end
    return mean(F)
end


"""
Computes the similarity between recurrence plots `RP₁` and `RP₂`. Outputs the
fraction of recurrences rates gained from RP₁ and of the joint recurrence
plot `RP₁ .* RP₂`.
"""
function jrp_rr_frac(RP₁::RecurrenceMatrix, RP₂::RecurrenceMatrix)
    @assert size(RP₁) == size(RP₂)

    RR1 = sum(RP₁)/(size(RP₁,1)*size(RP₁,1))
    JRP = elementwise_product(RP₁, RP₂)
    RR2 = sum(JRP)/(size(JRP,1)*size(JRP,1))

    f = RR2 / RR1
    return f
end

"""
Compute elementwise product of two Recurrence Plots (==JRP)
"""
elementwise_product(RP₁, RP₂) = JointRecurrenceMatrix(RP₁, RP₂)

