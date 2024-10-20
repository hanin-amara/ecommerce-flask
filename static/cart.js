document.addEventListener("DOMContentLoaded", function() {
    const addToCartBtn = document.querySelector('.btn');
    const cartTableBody = document.querySelector("#cart-table tbody");

    // Fonction pour rendre le panier
    function renderCart() {
        let cart = JSON.parse(localStorage.getItem('cart')) || [];
        cartTableBody.innerHTML = ''; // Effacer les articles précédents du panier

        cart.forEach((item, index) => {
            let row = `
                <tr class="${item.confirmed ? 'confirmed' : ''}"> <!-- Classe pour le style des articles confirmés -->
                    <td><img src="${item.image}" alt="${item.name}" width="50"></td>
                    <td>${item.name}</td>
                    <td>${item.color}</td>
                    <td>${item.size}</td>
                    <td>$${item.price.toFixed(2)}</td> <!-- Prix unitaire -->
                    <td>$${(item.price * item.quantity).toFixed(2)}</td> <!-- Prix total pour cet article -->
                    <td>
                        <input type="number" value="${item.quantity}" min="1" data-index="${index}" class="quantity-input"${item.confirmed ? ' disabled' : ''}>
                        ${!item.confirmed ? `
                            <button class="update-btn" data-index="${index}">🔄</button>
                            <button class="confirm-btn" data-index="${index}">✅</button>
                        ` : ''}
                        <button class="remove-btn" data-index="${index}">❌</button> <!-- Bouton pour retirer tous les produits -->
                    </td>
                </tr>
            `;
            cartTableBody.insertAdjacentHTML('beforeend', row);
        });

        // Aucune ligne pour le total général

        updateFloatingCart(); // Mettre à jour le panier flottant après le rendu
    }

    // Fonction pour mettre à jour l'icône du panier flottant
    function updateFloatingCart() {
        let cart = JSON.parse(localStorage.getItem('cart')) || [];
        let totalQuantity = 0;

        cart.forEach(item => {
            totalQuantity += item.quantity;
        });

        document.querySelector('.cart-quantity').innerText = totalQuantity;
        document.querySelector('.cart-total').innerText = `$${calculateTotal().toFixed(2)}`; // Mettre à jour l'affichage du prix total
    }
// Function to calculate total price
function calculateTotal() {
    let cart = JSON.parse(localStorage.getItem('cart')) || [];
    let totalPrice = 0;

    cart.forEach(item => {
        totalPrice += item.price * item.quantity;
    });

    return totalPrice;
}

    // Rendu initial du panier
    renderCart();

    // Fonctionnalité Ajouter au Panier
    addToCartBtn.addEventListener('click', function(event) {
        event.preventDefault();

        const productName = document.querySelector('h1').innerText;
        const productPrice = parseFloat(document.querySelector('.price').innerText.replace('$', ''));
        const selectedSize = document.getElementById('size').value;
        const selectedColor = document.querySelector('.color.active').dataset.color;
        const productImage = document.querySelector('.color.active img').src;

        if (selectedSize === "Select") {
            alert('Veuillez sélectionner une taille');
            return;
        }

        let product = {
            name: productName,
            price: productPrice,
            size: selectedSize,
            color: selectedColor,
            image: productImage,
            quantity: 1,
            confirmed: false // Suivre si le produit est confirmé
        };

        let cart = JSON.parse(localStorage.getItem('cart')) || [];
        cart.push(product);
        localStorage.setItem('cart', JSON.stringify(cart));

        renderCart();
    });

    // Écouteur d'événements pour mettre à jour, retirer, confirmer et annuler des articles
    cartTableBody.addEventListener('click', function(event) {
        let target = event.target;
        let cart = JSON.parse(localStorage.getItem('cart')) || [];

        // Retirer un article du panier
        if (target.classList.contains('remove-btn')) {
            let index = target.getAttribute('data-index');
            cart.splice(index, 1);
            localStorage.setItem('cart', JSON.stringify(cart));
            renderCart();
        }

        // Mettre à jour la quantité d'un article dans le panier
        if (target.classList.contains('update-btn')) {
            let index = target.getAttribute('data-index');
            let quantityInput = document.querySelector(`.quantity-input[data-index="${index}"]`);
            let newQuantity = parseInt(quantityInput.value);
            cart[index].quantity = newQuantity;
            localStorage.setItem('cart', JSON.stringify(cart));
            renderCart();
        }

        // Confirmer l'action d'un article dans le panier
        if (target.classList.contains('confirm-btn')) {
            let index = target.getAttribute('data-index');
            cart[index].confirmed = true; // Marquer le produit comme confirmé
            localStorage.setItem('cart', JSON.stringify(cart)); // Enregistrer les modifications

            renderCart(); // Re-rendre le panier pour refléter les changements
        }

        // Annuler l'action d'un article confirmé dans le panier
        if (target.classList.contains('cancel-btn')) {
            let index = target.getAttribute('data-index');
            cart[index].confirmed = false; // Marquer le produit comme non confirmé
            localStorage.setItem('cart', JSON.stringify(cart)); // Enregistrer les modifications

            renderCart(); // Re-rendre le panier pour refléter les changements
        }
    });
});
