import tensorflow as tf

def verify_gradient_flow(model, sample_batch):
    """
    Asserts that gradients are computed, non-None, and non-zero
    for both the Query and Candidate towers.
    """
    with tf.GradientTape(persistent=True) as tape:
        query_emb, candidate_emb = model(sample_batch, training=True)
        loss, _, _ = model.compute_loss(query_emb, candidate_emb)

    # 1. Check Query Tower Gradients
    query_vars = model.query_tower.trainable_variables
    query_grads = tape.gradient(loss, query_vars)
    for var, grad in zip(query_vars, query_grads):
        assert grad is not None, f"Gradient is None for Query variable: {var.name}"
        assert tf.reduce_sum(tf.abs(grad)) > 0, f"Zero gradient detected in Query variable: {var.name}"

    # 2. Check Candidate Tower Gradients
    candidate_vars = model.candidate_tower.trainable_variables
    candidate_grads = tape.gradient(loss, candidate_vars)
    for var, grad in zip(candidate_vars, candidate_grads):
        assert grad is not None, f"Gradient is None for Candidate variable: {var.name}"
        assert tf.reduce_sum(tf.abs(grad)) > 0, f"Zero gradient detected in Candidate variable: {var.name}"

    print("✅ Gradient check passed: Both Query and Candidate towers receive non-zero gradients!")