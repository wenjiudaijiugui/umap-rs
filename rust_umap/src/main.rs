use rust_umap::{UmapModel, UmapParams};

fn make_toy_data(n: usize) -> Vec<Vec<f32>> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;

        let x1 = (2.0 * std::f32::consts::PI * t).cos();
        let y1 = (2.0 * std::f32::consts::PI * t).sin();

        let x2 = 1.0 - (2.0 * std::f32::consts::PI * t).cos();
        let y2 = -0.5 - (2.0 * std::f32::consts::PI * t).sin() * 0.7;

        if i % 2 == 0 {
            data.push(vec![x1, y1]);
        } else {
            data.push(vec![x2, y2]);
        }
    }
    data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = make_toy_data(200);

    let params = UmapParams {
        n_epochs: Some(200),
        ..UmapParams::default()
    };

    let mut umap = UmapModel::new(params);
    let embedding = umap.fit_transform(&data)?;

    if let Some((a, b)) = umap.ab_params() {
        println!("learned curve parameters: a={a:.6}, b={b:.6}");
    }

    println!(
        "embedding shape: {} x {}",
        embedding.len(),
        embedding[0].len()
    );
    println!("first 10 embedded points:");
    for (i, row) in embedding.iter().take(10).enumerate() {
        println!("{i:3}: [{:.6}, {:.6}]", row[0], row[1]);
    }

    let query = data.iter().skip(190).cloned().collect::<Vec<_>>();
    let transformed = umap.transform(&query)?;
    println!(
        "transformed shape: {} x {}",
        transformed.len(),
        transformed[0].len()
    );

    let inverse = umap.inverse_transform(&transformed)?;
    println!(
        "inverse transformed shape: {} x {}",
        inverse.len(),
        inverse[0].len()
    );

    Ok(())
}
