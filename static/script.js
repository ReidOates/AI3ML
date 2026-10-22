
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const button = document.getElementById("submit-button");

    if (!form || !button) return;

    form.addEventListener("submit", () => {
        if (!form.checkValidity()) return;

        button.disabled = true;
        button.querySelector("span:first-child").textContent =
            "Memproses prediksi...";
    });
});
