// Sélectionner les éléments nécessaires
const colors = document.querySelectorAll('.color');
const colorLabel = document.querySelector('.color-selection label');
const mainImg = document.getElementById('MainImg');

// Ajout d'un écouteur d'événements sur chaque couleur
colors.forEach(color => {
    color.addEventListener('click', function() {
        // Supprimer la classe active de toutes les couleurs
        colors.forEach(c => c.classList.remove('active'));

        // Ajouter la classe active à la couleur sélectionnée
        this.classList.add('active');

        // Mettre à jour le label avec la couleur sélectionnée
        const selectedColor = this.getAttribute('data-color');
        colorLabel.textContent = `Colour: ${selectedColor}`;

        // Mettre à jour l'image principale du produit
        const selectedImg = this.getAttribute('data-img');
        mainImg.src = selectedImg;
    });
});

