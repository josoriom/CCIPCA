'use strict';

const { Matrix } = require('ml-matrix');

/**
 * Creates new Candid covariance-free incremental PCA
 * @param {Matrix} dataset - dataset or covariance matrix
 * @param {Object} [options]
 * @param {number} [options.l = 0] - By default (l = 0) stationary case. Nonstationary processes, should typically range between 2 and 4.
 * */

class CCIPCA {
  constructor(x, options = {}) {
    const { l = 0 } = options;
    if (l) {
      this.l = l;
    } else {
      this.l = 0;
    }
    let X = x.clone().center('column');
    let Xm = new Matrix(X.rows + 1, X.columns).setRow(0, X.getRowVector(0));
    for (let i = 1; i < X.rows; i++) {
      Xm.getRowVector(i).add(X.getRowVector(i));
    }
    Xm.getRowVector(0).div(X.rows);
    for (let i = 1; i < X.rows + 1; i++) {
      Xm.setRow(i, Xm.getRowVector(i - 1)
        .mul(i / (i + 1))
        .add((X.getRowVector(i - 1))
          .mul(1 / i)));
    }
    let Xc = new Matrix(X.rows, X.columns);
    for (let i = 0; i < X.rows; i++) {
      Xc.setRow(i, X.getRowVector(i).sub(Xm.getRowVector(i + 1)));
    }

    let v = new Matrix(Xc.rows, Xc.columns).setRow(0, X.getRowVector(0));
    for (let n = 1; n < Xc.rows; n++) {
      v.setRow(n, v.getRowVector(n - 1)
        .mul((n - this.l) / (n + 1))
        .add((v.getRowVector(n - 1)
          .div(v.getRowVector(n - 1).norm()))
          .mmul(Xc.getRowVector(n).transpose()
            .mmul(Xc.getRowVector(n)))
          .mul((1 + this.l) / (n + 1))));
    }
    let evector = new Matrix(Xc.columns, Xc.columns);
    let evalues = [];
    evector.setColumn(0, v.getRowVector(Xc.rows - 1)
      .transpose()
      .div(v.getRowVector(Xc.rows - 1).norm()));
    evalues.push(v.getRowVector(Xc.rows - 1).norm());


    for (let i = 1; i < Xc.columns; i++) {
      for (let n = 0; n < Xc.rows; n++) {
        Xc.setRow(n, Xc.getRowVector(n)
          .sub(Xc.getRowVector(n)
            .mmul(v.getRowVector(Xc.rows - 1)
              .transpose()
              .div(v.getRowVector(Xc.rows - 1)
                .norm()))
            .mmul(v.getRowVector(Xc.rows - 1)
              .div(v.getRowVector(Xc.rows - 1)
                .norm()))).transpose());
      }
      v.setRow(0, Xc.getRowVector(0));
      for (let n = 1; n < Xc.rows; n++) {
        v.setRow(n, v.getRowVector(n - 1)
          .mul((n - this.l) / (n + 1))
          .add((v.getRowVector(n - 1)
            .div(v.getRowVector(n - 1)
              .norm()))
            .mmul(Xc.getRowVector(n)
              .transpose()
              .mmul(Xc.getRowVector(n)))
            .mul((1 + this.l) / (n + 1))));
      }
      evector.setColumn(i, v.getRowVector(Xc.rows - 1)
        .transpose()
        .div(v.getRowVector(Xc.rows - 1).norm()));
      evalues.push(v.getRowVector(Xc.rows - 1).norm());
    }
    this.evector = evector;
    this.evalues = evalues;
  }

  /**
   * Returns the Eigenvectors
   * @returns {Matrix}
   */
  getEigenvectors() {
    return this.evector;
  }

  /**
   * Returns the Eigenvalues (on the diagonal)
   * @returns {[number]}
   */
  getEigenvalues() {
    return this.evalues;
  }

  /**
   * Returns the standard deviations of the principal components
   * @returns {[number]}
   */
  getStandardDeviations() {
    return this.evalues.map((x) => Math.sqrt(x));
  }
}
module.exports = CCIPCA;
